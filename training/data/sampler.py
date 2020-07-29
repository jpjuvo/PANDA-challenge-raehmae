import torch
import os
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from data.tileimages import *
from data.multitask import *
import fastai
from fastai.vision import *

class FoldSampler:

    def __init__(self, TRAIN, LABELS,
                 mean, std, N, 
                 tfms=[], sz=128,bs=16, 
                 n_folds=4, uniform_augmentations=False,
                 shuffle_nonempty_imgs=False,
                 model_name=None,
                 is_train=True,
                 is_ordinal=False,
                 SEED=2020,
                 num_workers=4):
        
        self._seed_everything(SEED)
        self.SEED = SEED
        self.tfms = tfms
        self.mean = mean
        self.std = std
        self.N = N
        self.nfolds = n_folds
        self.TRAIN = TRAIN
        self.sz = sz
        self.bs = bs
        self.is_ordinal = is_ordinal
        self.is_train=is_train
        self.num_workers=num_workers
        self.model_name = model_name
        self.uniform_augmentations = uniform_augmentations
        self.shuffle_nonempty_imgs = shuffle_nonempty_imgs
        self._prepare_data(TRAIN, LABELS)
        self.df.head()

    def _seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _cats4slide(self, image_id):
        fn_cats = os.path.join(self.TRAIN, f'{image_id}_mask.txt')
        if os.path.isfile(fn_cats):
            with open(fn_cats) as f:
                return [int(int(l)>1) for l in f.readlines()]
        else:
            raise Exception("File not found", str(fn_cats))

    def _findAllReplicates(self, pairs, seed):
        replicates = [seed]
        nodes = [seed]
        
        def addReplicate(n):
            if n not in replicates:
                replicates.append(n)
                nodes.append(n)
        
        # while there are nodes left
        while len(nodes) > 0:
            this_node = nodes[0]
            for i,j in pairs:
                if i==this_node:
                    # match - add j to replicates
                    addReplicate(j)
                elif j==this_node:
                    # match - add i to replicates
                    addReplicate(i)
            nodes.pop(0)
        return replicates

    def _pairs2sameFolds(self, df,pairs):
        replicate_indices = np.unique(pairs)
        split_values = df.split.values
        for ind in replicate_indices:
            allReps = self._findAllReplicates(list(pairs), ind)
            # set all to the same fold as the minimum index
            min_rep = min(allReps)
            target_fold = split_values[min_rep]
            for rep in allReps:
                split_values[rep] = target_fold
        df.split = split_values
        return df

    def _prepare_data(self, TRAIN, LABELS):
        df = pd.read_csv(LABELS).set_index('image_id')
        files = set([p[:32] for p in os.listdir(TRAIN)])
        df = df.loc[files]
        df = df.reset_index()
        df['stratify'] = df.data_provider.map(str) + '-' +  df.isup_grade.map(str)
        splits = StratifiedKFold(n_splits=self.nfolds, random_state=self.SEED, shuffle=True)
        splits = list(splits.split(df,df.stratify))
        folds_splits = np.zeros(len(df)).astype(np.int)
        for i in range(self.nfolds): folds_splits[splits[i][1]] = i
        df['split'] = folds_splits

        if self.is_ordinal:
            def _transform_ordinal(label):
                #return ','.join([str(i) for i in range(int(label) + 1)])
                return ','.join([str(i) for i in range(int(label))])
            df.isup_grade = df.isup_grade.apply(_transform_ordinal)

        # add tile cancer categories if present in train data
        if self.model_name in ["multihead_tilecat", "multihead_tilecat_attention"]:
            cancer_labels = np.array([np.array(self._cats4slide(image_id)) for image_id in df.image_id.values])
            for i in range(cancer_labels.shape[1]):
                df[f'cancer_status_{i}'] = list(cancer_labels[:,i])

        # set serial section replicates to same folds
        pairs_fn = os.path.join('../','pair_indices.npy')
        if os.path.exists(pairs_fn):
            pairs = np.load(pairs_fn)
            print(f'Setting {np.array(pairs).shape[0]} serial section replicates to same folds')
            df = self._pairs2sameFolds(df, pairs)

        self.df = df

    def get_data(self,fold=0, **kwargs):
        model_name = "iafoss" if self.model_name is None else self.model_name
        regr = "regr" in model_name

        def __MImageItemList():
            """ This returns MImageItemList with specified defaults """
            return MImageItemList.from_df(self.df,
                                            path='.',
                                            folder=self.TRAIN,
                                            cols='image_id',
                                            sz=self.sz,
                                            N=self.N,
                                            mean=self.mean,
                                            std=self.std,
                                            uniform_augmentations=self.uniform_augmentations,
                                            shuffle_nonempty_imgs=self.shuffle_nonempty_imgs
                                        )
        if model_name in ["multihead_tilecat", "multihead_tilecat_attention"] and self.is_train:
            # create isup LabelItemList
            isup_labels = (
                (__MImageItemList()
                    .split_by_idx(self.df.index[self.df.split == fold].tolist())
                    .label_from_df(cols=['isup_grade']))
            )
            # create the dict to hold all LabelItemLists
            multitask_project = {
                'isup': {
                    'label_lists': isup_labels,
                }
            }
            # add tile cancer categories to the dict
            for i in range(self.N):
                tilecat = (__MImageItemList()
                    .split_by_idx(self.df.index[self.df.split == fold].tolist())
                    .label_from_df(cols=[f'cancer_status_{i}']))
                multitask_project[f'tilecat_{i}'] = {
                    'label_lists': tilecat,
                }

            ItemLists.label_from_mt_project = label_from_mt_project
            return (__MImageItemList()
                    .split_by_idx(self.df.index[self.df.split == fold].tolist())
                    .label_from_mt_project(multitask_project)
                    .transform(self.tfms,
                            size=self.sz,
                            padding_mode='zeros')
                    .databunch(bs=self.bs,
                            num_workers=self.num_workers)
                    )
        else: # Defaults to Iafoss
            if self.is_ordinal:
                return (__MImageItemList()
                        .split_by_idx(self.df.index[self.df.split == fold].tolist())
                        .label_from_df(cols=['isup_grade'], label_cls=None, label_delim=',')
                        .transform(self.tfms,
                                size=self.sz,
                                padding_mode='zeros')
                        .databunch(bs=self.bs,
                                num_workers=self.num_workers)
                    )
            else:
                return (__MImageItemList()
                        .split_by_idx(self.df.index[self.df.split == fold].tolist())
                        .label_from_df(cols=['isup_grade'], label_cls=FloatList if regr==True else None)
                        .transform(self.tfms,
                                size=self.sz,
                                padding_mode='zeros')
                        .databunch(bs=self.bs,
                                num_workers=self.num_workers)
                    )
