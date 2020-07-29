import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from train.mish_activation import *

class Model(nn.Module):
    def __init__(self, enc_func, n=6, final_dropout=0.5, tile_list_input=True, pre=True):
        super().__init__()
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
            Mish(),
            nn.BatchNorm1d(512), 
            nn.Dropout(final_dropout),
            nn.Linear(512,1),
#             Mish(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(final_dropout),
#             nn.Linear(128,1)
            
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
        
        #x: bs*N x 3 x IMG_SZ x IMG_SZ
        x = self.enc(x)
        
        #x: bs*N x 1280 x 7 x 7
        shape = x.shape
        
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        
        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        return x
