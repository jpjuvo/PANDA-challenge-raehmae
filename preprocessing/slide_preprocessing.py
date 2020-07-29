import pandas as pd
import numpy as np

from skimage.io import MultiImage
from skimage.morphology import skeletonize
import maskslic as seg

import cv2
import matplotlib.pyplot as plt

from pathlib import Path

import warnings

VALID_SLIDE_EXTENSIONS = {'.tiff', '.mrmx', '.svs'}


# ~~~~~~~~~~~~ Helper functions ~~~~~~~~~~~~
def generateMetaDF(data_dir, meta_fn:str='train.csv'):
    '''
        Makes a pandas.DataFrame of paths out of a directory including slides. Drop the `train.csv` in `data_dir`
        and the script will also merge any meta data from there on `image_id` key.
    '''
    
    
    all_files = [path.resolve() for path in Path(data_dir).rglob("*.*")]
    slide_paths = [path for path in all_files if path.suffix in VALID_SLIDE_EXTENSIONS]
    
    if len(slide_paths)==0:
        raise ValueError('No slides in `data_dir`=%s'%data_dir)
    
    data_df = pd.DataFrame({'slide_path':slide_paths})
    data_df['image_id'] = data_df.slide_path.apply(lambda x: x.stem)
    
    slides = data_df[~data_df.image_id.str.contains("mask")]
    masks = data_df[data_df.image_id.str.contains("mask")]
    masks['image_id'] = masks.image_id.str.replace("_mask", "")
    masks.columns = ['mask_path', 'image_id']
    
    data_df = slides.merge(masks, on='image_id', how='left')
    data_df['slide_path'] = data_df.slide_path.apply(lambda x: str(x) if not pd.isna(x) else None)
    data_df['mask_path'] = data_df.mask_path.apply(lambda x: str(x) if not pd.isna(x) else None)
    

    ## Merge metadata
    meta_csv = [file for file in all_files if file.name==meta_fn]
    if meta_csv:
        meta_df = pd.read_csv(str(meta_csv[0]))
        data_df = data_df.merge(meta_df, on='image_id')
    
    return data_df

def tileClassification(tile, provider:str):
    '''
        Returns the cancer class of a tile based on majority vote of the tile's annotated pixels
            0: background (non tissue) or unknown
            1: benign tissue (stroma and epithelium combined)
            2: cancerous tissue (stroma and epithelium combined)
    '''

    if provider == "karolinska":
        '''
        Karolinska: Regions are labelled. Valid values are:
            0: background (non tissue) or unknown
            1: benign tissue (stroma and epithelium combined)
            2: cancerous tissue (stroma and epithelium combined)
        '''
        classes = {0:0, 1:1, 2:2}    
        
    elif provider == "radboud":
        ''' 
        Radboud: Prostate glands are individually labelled. Valid values are:
            0: background (non tissue) or unknown
            1: stroma (connective tissue, non-epithelium tissue)
            2: healthy (benign) epithelium
            3: cancerous epithelium (Gleason 3)
            4: cancerous epithelium (Gleason 4)
            5: cancerous epithelium (Gleason 5)
        '''
        classes = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2}
    
    tile = np.int32(tile)
    counts = np.bincount(tile.reshape(-1,1)[:,0])

    ## If only background, accept background (class=0) as the annotation
    if len(counts)==1:
        return 0

    ## Otherwise take the second most common pixel as the annotation
    max_annotation = np.argmax(counts[1:])       
    
    return classes[max_annotation+1]
    
def getTopLeftCorners(dims, tile_size:int):
    ''' Make a map of the tiles' locations '''
    
    cols, rows = np.ceil(dims/tile_size).astype(int)

    ## M
    top_left_corners = []
    for i in range(cols):
        for j in range(rows):
            top_left_corners.append( (i*tile_size, j*tile_size) )
            
    return np.array(top_left_corners)

def makeTissueMask(img):
    ''' Makes a tissue mask. Also filters the green / blue pen markings '''

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Filter green/blue ink marker annotations
    lower_red = np.array([120,20,180])
    upper_red = np.array([190,220,255])
    tissue_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Post-process
    tissue_mask = cv2.dilate(tissue_mask, None, iterations=2)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, None)
    tissue_mask = cv2.medianBlur(tissue_mask, 21)
    
    return tissue_mask

def fillTissueMask(tissue_mask):
    ''' Fills the holes in a tissue mask by filling the corresponding contours '''

    new_tm = tissue_mask.copy()
    contours,_ = cv2.findContours(new_tm,0,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(new_tm,contours,-1,255,-1)
    
    return new_tm

def estimateWithCircles(arch, radius:int=64):

    arch = arch.squeeze()
    circle_centers=[arch[0]]
    for point in arch:
        if np.all(np.linalg.norm(point-np.array(circle_centers),axis=-1)>radius):
            circle_centers.append(point)
    return np.array(circle_centers)

def getTissuePercentages(tissue_mask, level_offset:int, tile_size:int, top_left_corners):
    ''' Calculate the tissue percentage per each tile. Tissue ~ non 255 on the red-channel '''

    ds_rate = 4**level_offset
    tissue_pcts = np.array([tissue_mask[j//ds_rate:(j+tile_size)//ds_rate,
                                        i//ds_rate:(i+tile_size)//ds_rate].sum()
                            for (i,j) in top_left_corners])/(255*(tile_size/ds_rate)**2)
    
    return tissue_pcts

def padIfNeeded(img, tgt_width:int=128, tgt_height:int=128, border_color:int=255):
    ''' Pad images that need padding (padding on right and bottom)'''
    
    h,w = img.shape[0:2]
    
    if w<tgt_width or h<tgt_height:
        padded = np.ones((tgt_height,tgt_width,3), dtype='uint8')*border_color
        padded[:h,:w] = img
        return padded

    return img

def distributeIntToChunks(available:int, weights):
    '''
        To distribute `available` "seats" based on weights, see
        https://stackoverflow.com/questions/9088403/distributing-integers-using-weights-how-to-calculate
    '''
    
    distributed_amounts = []
    total_weights = sum(weights)
    for weight in weights:
        weight = float(weight)
        p = weight / total_weights
        distributed_amount = round(p * available)
        distributed_amounts.append(distributed_amount)
        total_weights -= weight
        available -= distributed_amount
    return np.int32(distributed_amounts)


# ~~~~~~~~~~~~ Slide class ~~~~~~~~~~~~
class Slide:
    
    def __init__(self,slide_fn, level=2, tile_size=128, mask_fn=None, data_provider=None):
        self.slide_fn = slide_fn
        self.level = level
        self.tile_size = tile_size

        self.img = MultiImage(self.slide_fn)[self.level].copy()
        self.dims = np.array(self.img.shape[:2][::-1])
        self.ds_img = MultiImage(self.slide_fn)[2].copy()
        self.tissue_mask = makeTissueMask(self.ds_img)


        self.mask_fn = mask_fn
        self.data_provider = data_provider
        if not (self.mask_fn==None or self.data_provider==None):
            self.mask = MultiImage(mask_fn)[level].sum(axis=-1)

        self.tile_coords = None


    def getTileCoords(self, num_tiles:int, sampling_method='skeleton', tissue_th:tuple=(0.2, 0.7), seed=None, offset:tuple=(0,0)):
        ''' 
            Find `num_indices` indices with maximal amount of tissue. `tissue_th`  is the slice of tissue percentage ~(min, max)
            within which we allow the search: sometimes the `max` might not give enough tiles 
            for the mosaic, so we can decrease it gradually until `min` is reached. 

            offset = (x,y) coord offset in tile size fractions that is applied to sampled coords. This is meant as a form of data augmentation.
        '''

        assert sampling_method in {'skeleton', 'tissue_pct', 'slic'}
        "`sampling_method` should be one of 'skeleton', 'tissue_pct' or 'slic'"
        
        level_offset = 2 - self.level
        
        if sampling_method == 'skeleton':

            # Determine skeleton from filled tissue mask
            filled_tissue_mask = fillTissueMask(self.tissue_mask.copy())
            filled_tissue_mask = filled_tissue_mask // 255  # needs to be an array of 0s and 1s
            skeleton = skeletonize(filled_tissue_mask, method='lee')
            self.skeleton = np.uint8(np.where(skeleton != 0, 255, 0))

            skeleton = cv2.dilate(self.skeleton, None)
            contours, _ = cv2.findContours(skeleton, 0, cv2.CHAIN_APPROX_NONE)

            # Filter contours based on length
            arch_lens = []
            valid_indices = []
            radius=int( self.tile_size / (4 ** level_offset) )
            for idx, arch in enumerate(contours):
                c = cv2.arcLength(arch, False)
                c = c / 2
                if c < radius / 4:
                    continue

                valid_indices.append(idx)
                arch_lens.append(c)

            if not np.array(valid_indices).size == 0:
                contours = np.array(contours)[valid_indices]
            else: #if no main skeletons were found, it could be that the tissue slide is just very small
                arch_lens = [1/len(contours)]*len(contours)

            # Extract points from the accepted contours
            weights = np.array(arch_lens) / np.sum(arch_lens)
            points_per_arch = distributeIntToChunks(num_tiles, weights)  # <- number of points to be extracted
            for idx, arch in enumerate(contours):

                num_indices = points_per_arch[idx]
                output = np.zeros_like(skeleton)
                cv2.drawContours(output, [arch], -1, 1, 1)

                y_, x_ = np.where(output)

                # Simplify the shape by fitting circles
                arch = np.dstack([x_, y_])
                cps = estimateWithCircles(arch, radius)
                cx, cy = cps[..., 0], cps[..., 1]  #

                # Randomly select indices, in case too many; seed if needed
                if len(cx) > num_indices:

                    # Seed if needed
                    if not seed == None:
                        np.random.seed(seed)
                    indices = sorted(np.random.choice(len(cx), points_per_arch[idx], replace=False))
                    np.random.seed(None)  # Return clock seed

                    cx, cy = cx[indices], cy[indices]

                # Append to returnables
                if idx == 0:
                    intermediate_coords = np.dstack([cx, cy])
                else:
                    intermediate_coords = np.hstack([intermediate_coords, np.dstack([cx, cy])])

            # To top-left-corner format
            final_coords = intermediate_coords.squeeze()*4**level_offset
            if final_coords.shape == (2,):
                final_coords = np.expand_dims(final_coords, 0)

            final_coords = final_coords - np.array( [self.tile_size//2,self.tile_size//2])

        elif sampling_method == 'tissue_pct':
            self.top_left_corners = getTopLeftCorners(self.dims, self.tile_size)
            self.tissue_pcts = getTissuePercentages(self.tissue_mask, level_offset=level_offset,
                                                    tile_size=self.tile_size,
                                                    top_left_corners=self.top_left_corners)

            # Find indices
            tth_min, tth = tissue_th
            while len(np.where(self.tissue_pcts > tth)[0]) < num_tiles:
                if tth <= tth_min:
                    break

                tth -= 0.05

            # Indices
            indices = np.where(self.tissue_pcts > tth)[0]

            # Randomly select indices, in case too many; seed if needed
            if len(indices)>num_tiles:
                if not seed == None:
                    np.random.seed(seed)
                indices = sorted(np.random.choice(indices, num_tiles, replace=False))
                np.random.seed(None)  # Return clock seed

            final_coords = self.top_left_corners[indices].copy()

        elif sampling_method == 'slic':

            # Determine SLIC clusters from filled tissue mask
            filled_tissue_mask = fillTissueMask(self.tissue_mask.copy())
            filled_tissue_mask = filled_tissue_mask // 255  # needs to be an array of 0s and 1s

            segments = seg.slic(self.ds_img, compactness=10, seed_type='nplace', mask=filled_tissue_mask, n_segments=num_tiles,
                                multichannel=True, recompute_seeds=True, enforce_connectivity=True)
            indices = [k for k in np.unique(segments) if not k==-1]

            # Randomly select indices, in case too many; seed if needed
            if len(indices)>num_tiles:
                if not seed == None:
                    np.random.seed(seed)
                indices = sorted(np.random.choice(indices, num_tiles, replace=False))
                np.random.seed(None)  # Return clock seed

            for i in indices:
                contours, _ = cv2.findContours(np.uint8(np.where(segments==i, 255, 0)), 0, 1)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x))[::-1]
                
                M = cv2.moments(contours[0])
                cx = np.int32(M['m10'] / M['m00'])
                cy = np.int32(M['m01'] / M['m00'])

                if i==0:
                    intermediate_coords = np.dstack([cx, cy])
                else:
                    intermediate_coords = np.hstack([intermediate_coords, np.dstack([cx, cy])]) 
                           
            # Append more cluster contours if num_tiles has not been reached
            if len(indices)<num_tiles:
                enough_tiles = False
                for i in indices:
                    contours, _ = cv2.findContours(np.uint8(np.where(segments==i, 255, 0)), 0, 1)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[::-1]
                    
                    for j, cnt in enumerate(contours):
                        # accept a slic cluster contour if it's area is at least 5% of tile area
                        if j!=0 and cv2.contourArea(cnt)>(0.05 * self.tile_size / (4**level_offset))**2:
                            M = cv2.moments(cnt)
                            cx = np.int32(M['m10'] / M['m00'])
                            cy = np.int32(M['m01'] / M['m00'])

                            intermediate_coords = np.hstack([intermediate_coords, np.dstack([cx, cy])])
                            
                            if intermediate_coords.shape[1] == 12:
                                enough_tiles = True
                                break
                    if enough_tiles:
                        break
                            
            # To top-left-corner format
            final_coords = intermediate_coords.squeeze() * 4 ** level_offset
            final_coords = final_coords - np.array([self.tile_size // 2, self.tile_size // 2])

        # apply offset to coordinates
        final_coords = final_coords + np.array( [int(self.tile_size*offset[0]),int(self.tile_size*offset[1])])

        self.tile_coords = final_coords.copy()


    def getTiles(self, stack:bool=False, sampling_method:str='skeleton', mosaic_grid:tuple=(4,3), output_tile_size:int=128, tissue_th:tuple=(0.1, 0.7), seed:int=None, offset:tuple=(0,0)):
        ''' Get tiles from the slide and stack into mosaic if needed 
        
        offset = (x,y) coord offset in tile size fractions that is applied to sampled coords. This is meant as a form of data augmentation.
        '''

        # Solve indices to be used in mosaic
        m, n = mosaic_grid
        self.getTileCoords(num_tiles=n*m, sampling_method=sampling_method, tissue_th=tissue_th, seed=seed, offset=offset)

        # Read regions
        output_img = np.ones([n * m, output_tile_size, output_tile_size, 3], dtype='uint8') * 255

        for idx, coord in enumerate(self.tile_coords):
            x, y = coord
            left, right = np.int32(np.clip([x, x + self.tile_size], 0, self.dims[0]))
            bottom, top = np.int32(np.clip([y, y + self.tile_size], 0, self.dims[1]))

            tile = self.img[bottom:top, left:right].copy()
            tile = padIfNeeded(tile, tgt_width=self.tile_size, tgt_height=self.tile_size)
            tile = cv2.resize(tile, (output_tile_size,) * 2)

            output_img[idx] = np.uint8(tile)

        if len(self.tile_coords)<m*n:
            warnings.warn("Could not find enough unique tiles for the slide"
                          "(tiles: %s/%s, slide: %s" %(len(self.tile_coords), m*n, self.slide_fn))

        # Stack to single array of (m,n) tiles if needed
        if stack:
            output_img = [np.hstack([output_img[i * n + j] for j in range(n)]) for i in range(m)]
            output_img = np.vstack(output_img)

        return np.array(output_img)

    def getTilesCancerStatus(self, stack:bool=False, mosaic_grid:tuple=(4,3)):
        ''' Get tiles from the slide and stack into mosaic if needed '''

        # Solve indices to be used in mosaic
        m, n = mosaic_grid

        # Read regions
        output_img = np.zeros([n * m], dtype='uint8')

        for idx, coord in enumerate(self.tile_coords):
            x, y = coord
            left, right = np.int32(np.clip([x, x + self.tile_size], 0, self.dims[0]))
            bottom, top = np.int32(np.clip([y, y + self.tile_size], 0, self.dims[1]))

            tile = self.mask[bottom:top, left:right].copy()
            tile_cat = tileClassification(tile, self.data_provider)

            output_img[idx] = tile_cat

        # Stack to single array of (m,n) tiles if needed
        if stack:
            output_img = [np.hstack([output_img[i * n + j] for j in range(n)]) for i in range(m)]
            output_img = np.vstack(output_img)

        return np.array(output_img)

    def visualizeCoverage(self, figsize=(16,16)):
        ''' Visualize the coverage of indices on a slide '''

        background = self.ds_img.copy()
        foreground = background.copy()

        level_offset = 2-self.level

        for idx, coord in enumerate(self.tile_coords):
            x, y = coord
            left, right = np.int32(np.clip([x, x + self.tile_size], 0, self.dims[0])/(4**level_offset))
            bottom, top = np.int32(np.clip([y, y + self.tile_size], 0, self.dims[1])/(4**level_offset))

            foreground[bottom:top, left:right] = (0, 255, 0)

        ## Visualize
        output = cv2.addWeighted(background, 0.7, foreground, 0.3, 0)
        
        plt.figure(figsize=figsize)
        plt.imshow(output)
        plt.show()

        