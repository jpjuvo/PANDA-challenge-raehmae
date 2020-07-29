import fastai
from fastai.vision import *
import random
import matplotlib.pyplot as plt
import PIL
from fastai.vision import Image
import random
import torch
import numpy as np

def open_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image, after_open:Callable=None)->Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        x = PIL.Image.open(fn).convert(convert_mode)
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(1.0-x) #invert image for zero padding

class MImage(ItemBase):
    def __init__(self, imgs, N, mean, std, sz, uniform_augmentations=False):
        self.N = N
        self.mean = mean
        self.std = std
        self.sz = sz
        self.uniform_augmentations = uniform_augmentations
        self.obj, self.data = \
          (imgs), [(imgs[i].data - mean[...,None,None])/std[...,None,None] for i in range(len(imgs))]
    
    def apply_tfms(self, tfms,*args, **kwargs):
        random_int = random.randint(0, 10000000) # for uniform augmentations
        random_state = random.getstate()
        for i in range(len(self.obj)):
            if self.uniform_augmentations:
                random.setstate(random_state) 
                torch.manual_seed(random_int)       
            self.obj[i] = self.obj[i].apply_tfms(tfms, *args, **kwargs)
            self.data[i] = (self.obj[i].data - self.mean[...,None,None])/self.std[...,None,None]
        return self
    
    def __repr__(self): return f'{self.__class__.__name__} {img.shape for img in self.obj}'
    def to_one(self):
        img = torch.stack(self.data,1)
        img = img.view(3,-1,self.N,self.sz,self.sz).permute(0,1,3,2,4).contiguous().view(3,-1,self.sz*self.N)
        return Image(1.0 - (self.mean[...,None,None]+img*self.std[...,None,None]))

class MImageItemList(ImageList):
    def __init__(self, N, sz, mean, std, uniform_augmentations=False,shuffle_nonempty_imgs=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.sz = sz
        self.mean = mean
        self.std = std
        self.uniform_augmentations = uniform_augmentations
        self.shuffle_nonempty_imgs = shuffle_nonempty_imgs
        self.get_iters = 0
        
        self.copy_new.append('N')
        self.copy_new.append('sz')
        self.copy_new.append('mean')
        self.copy_new.append('std')
        self.copy_new.append('uniform_augmentations')
        self.copy_new.append('shuffle_nonempty_imgs')
        
    def __len__(self)->int: return len(self.items) or 1 
    
    def get(self, i):
        fn = Path(self.items[i])
        fnames = [Path(str(fn)+'_'+str(i)+'.png') for i in range(self.N)]
        random.shuffle(fnames)
        imgs = [open_image(fname, convert_mode=self.convert_mode, after_open=self.after_open)
               for fname in fnames]
        if self.shuffle_nonempty_imgs:
            
            nonempty = [idx for idx,img in enumerate(imgs)
                        if not np.all(np.equal(np.array(img.data),
                                               np.zeros_like(np.array(img.data), dtype='float32'))
                                     )]
            empty = [k for k in range(len(imgs)) if not k in nonempty]
            self.get_iters +=1
            np.random.seed(self.get_iters)
            np.random.shuffle(nonempty)
            imgs = list(np.array(imgs)[nonempty]) + list(np.array(imgs)[empty])
            
        return MImage(imgs, self.N, self.mean, self.std, self.sz, self.uniform_augmentations)

    def reconstruct(self, t):
        return MImage([self.mean[...,None,None]+_t*self.std[...,None,None] for _t in t], self.N, self.mean, self.std, self.sz, self.uniform_augmentations)
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(300,50), **kwargs):
        rows = min(len(xs),8)
        fig, axs = plt.subplots(rows,1,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()
        
