import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from train.mish_activation import *

class Model(nn.Module):
    def __init__(self, enc_func, n=6,final_dropout=0.5, num_tiles=16, cancer_categories=2, is_train=True,  tile_list_input=True, pre=True):
        super().__init__()
        self.is_train = is_train
        self.tile_list_input = tile_list_input
        m = enc_func()
        # drop the final avgpool and fc layers
        self.enc = nn.Sequential(*list(m.children())[:-2])
        
        # get the output channels of the encoder by checking the input size of the fc layer
        # nc = 2048 with resnext50_32x4d_ssl
        nc = list(m.children())[-1].in_features

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
        
    def forward(self, *x):
        if self.tile_list_input:
            # x is N dimensional list of arrays of shape (bs,3,sz,sz)
            shape = x[0].shape
            N = len(x) # n is number of tiles per slide
            x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3]) #bs*N,3,sz,sz
        else:
            x = x[0] #untuple
            shape = x.shape
            n = shape[1]
            x = x.view(-1,shape[2],shape[3],shape[4]) #bs*N,3,sz,sz

        x = self.enc(x) #bs*N, C, 4, 4
        _, C, h, w = x.shape
        
        # cancer classification
        y = self.cancer_head(x) #y: bs*N, cancer_categories
        y = y.view(-1, N, y.shape[-1]) #y: bs, N, cancer_categories
        
        # attention
        x = x.permute(1,2,3,0)# x: C, h, w, bs*N
        mask = y.argmax(-1).view(-1)
        mask = (mask>0).half()
        x = (x*mask).permute(3,0,1,2)# x: bs*N, C,h,w 
        
        # concatenate the output for tiles into a single map - no matter how many tile images we have,
        # each slide is compressed to a max and avg of all tiles 
        x = x.view(-1,N,C,h,w).permute(0,2,1,3,4).contiguous() # x: bs, C, N, 4, 4
        x = x.view(-1,C,h*N,w) # x: bs, C, N*4, 4
        
        # isup classification
        x = self.head(x)
        
        outlist = [x] + [y[:,i,:] for i in range(y.shape[1])]
        return outlist if self.is_train else x