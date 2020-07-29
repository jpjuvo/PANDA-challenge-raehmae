import fastai
from fastai.vision import *
import random
import matplotlib.pyplot as plt
import PIL
from fastai.vision import Image
import random
import torch
import numpy as np

# Multitask implementation taken from
# https://nbviewer.jupyter.org/gist/denisvlr/802f980ff6b6296beaaea1a592724a51

def label_from_mt_project(self, multitask_project):
    mt_train_list = MultitaskItemList(
        [task['label_lists'].train.y for task in multitask_project.values()], 
        mt_names=list(multitask_project.keys(),
                    )
    )
    mt_valid_list = MultitaskItemList(
        [task['label_lists'].valid.y for task in multitask_project.values()], 
        mt_names=list(multitask_project.keys())
    )
    
    self.train = self.train._label_list(x=self.train, y=mt_train_list)
    self.valid = self.valid._label_list(x=self.valid, y=mt_valid_list)
    self.__class__ = MultitaskLabelLists # TODO: Class morphing should be avoided, to be improved.
    self.train.__class__ = MultitaskLabelList
    self.valid.__class__ = MultitaskLabelList
    return self

class MultitaskItem(MixedItem):    
    def __init__(self, *args, mt_names=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mt_names = mt_names
    
    def __repr__(self):
        return '|'.join([f'{self.mt_names[i]}:{item}' for i, item in enumerate(self.obj)])

class MultitaskItemList(MixedItemList):
    
    def __init__(self, *args, mt_names=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mt_classes = [getattr(il, 'classes', None) for il in self.item_lists]
        self.mt_types = [type(il) for il in self.item_lists]
        self.mt_lengths = [len(i) if i else 1 for i in self.mt_classes]
        self.mt_names = mt_names
        
    def get(self, i):
        return MultitaskItem([il.get(i) for il in self.item_lists], mt_names=self.mt_names)
    
    def reconstruct(self, t_list):
        items = []
        t_list = self.unprocess_one(t_list)
        for i,t in enumerate(t_list):
            if self.mt_types[i] == CategoryList:
                items.append(Category(t, self.mt_classes[i][t]))
            elif issubclass(self.mt_types[i], FloatList):
                items.append(FloatItem(t))
        return MultitaskItem(items, mt_names=self.mt_names)
    
    def analyze_pred(self, pred, thresh:float=0.5):         
        predictions = []
        start = 0
        for length, mt_type in zip(self.mt_lengths, self.mt_types):
            if mt_type == CategoryList:
                predictions.append(pred[start: start + length].argmax())
            elif issubclass(mt_type, FloatList):
                predictions.append(pred[start: start + length][0])
            start += length
        return predictions

    def unprocess_one(self, item, processor=None):
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: 
            item = _processor_unprocess_one(p, item)
        return item

def _processor_unprocess_one(self, item:Any): # TODO: global function to avoid subclassing MixedProcessor. To be cleaned.
    res = []
    for procs, i in zip(self.procs, item):
        for p in procs: 
            if hasattr(p, 'unprocess_one'):
                i = p.unprocess_one(i)
        res.append(i)
    return res

class MultitaskLabelList(LabelList):
    def get_state(self, **kwargs):
        kwargs.update({
            'mt_classes': self.mt_classes,
            'mt_types': self.mt_types,
            'mt_lengths': self.mt_lengths,
            'mt_names': self.mt_names
        })
        return super().get_state(**kwargs)

    @classmethod
    def load_state(cls, path:PathOrStr, state:dict) -> 'LabelList':
        res = super().load_state(path, state)
        res.mt_classes = state['mt_classes']
        res.mt_types = state['mt_types']
        res.mt_lengths = state['mt_lengths']
        res.mt_names = state['mt_names']
        return res
    
class MultitaskLabelLists(LabelLists):
    @classmethod
    def load_state(cls, path:PathOrStr, state:dict):
        path = Path(path)
        train_ds = MultitaskLabelList.load_state(path, state)
        valid_ds = MultitaskLabelList.load_state(path, state)
        return MultitaskLabelLists(path, train=train_ds, valid=valid_ds)
      