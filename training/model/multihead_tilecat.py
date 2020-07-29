import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from train.mish_activation import *

class Model(nn.Module):
    def __init__(self, enc_func, n=6,final_dropout=0.5, num_tiles=16, cancer_categories=3, is_train=True,  tile_list_input=True, pre=True):
        super().__init__()
        self.is_train = is_train
        self.tile_list_input = tile_list_input
        m = enc_func()
        # drop the final avgpool and fc layers
        self.enc = nn.Sequential(*list(m.children())[:-2])
        
        # get the output channels of the encoder by checking the input size of the fc layer
        # nc = 2048 with resnext50_32x4d_ssl
        nc = list(m.children())[-1].in_features
        
        # create the classification head, where input size is 2*nc and output n
        # input size is 2*nc because we use AdaptiveConcatPool
        # and it's a layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`, thus 2*nc
        # n=6 with isup_grades from 0 to 5
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2*nc,512),
            nn.BatchNorm1d(512),
            Mish(), 
            nn.Dropout(final_dropout),
            nn.Linear(512,n)
        )

        self.cancer_head = nn.Sequential(
          AdaptiveConcatPool2d(),
          Flatten(),
          nn.Linear(2*nc,512),
          nn.BatchNorm1d(512),
          Mish(), 
          nn.Dropout(final_dropout),
          nn.Linear(512,cancer_categories),
          nn.Softmax()
       )
        
    def forward(self, *x):
        if self.tile_list_input:
            # x.shape = (N,bs,3,sz,sz)
            shape = x[0].shape
            # n is number of tiles per slide
            n = len(x)
            # reshape x to a large batch of images
            x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        else:
            x=x[0] #untuple
            shape = x.shape
            n = shape[1]
            x = x.view(-1,shape[2],shape[3],shape[4])
        # x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        # x: bs*N x C x 4 x 4
        shape = x.shape
        # concatenate the output for tiles into a single map - no matter how many tile images we have,
        # each slide is compressed to a max and avg of all tiles 
        x1 = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
           .view(-1,shape[1],shape[2]*n,shape[3])
        # x: bs x C x N*4 x 4
        x1 = self.head(x1)
        
        x2 = self.cancer_head(x) #(bs*N, 3) 
        
        x2 = x2.view(-1, n, x2.shape[-1]) # bs, N, 3
        outlist = [x1] + [x2[:,i,:] for i in range(x2.shape[1])]
        return outlist if self.is_train else x1