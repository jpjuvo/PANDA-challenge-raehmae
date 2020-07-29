import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from train.mish_activation import *

class Model(nn.Module):
    def __init__(self, enc_func, n=6, final_dropout=0.5, pre=True, num_tiles=12, f_conv_out=128, tile_list_input=True):
        super().__init__()
        self.num_tiles = num_tiles
        self.f_conv_out = f_conv_out
        m = enc_func()

        self.tile_list_input = tile_list_input
        # drop the final avgpool and fc layers
        self.enc = nn.Sequential(*list(m.children())[:-2])
        
        # get the output channels of the encoder by checking the input size of the fc layer
        # nc = 2048 with resnext50_32x4d_ssl
        nc = list(m.children())[-1].in_features
        
        # Adaptive concat branch
        self.feat_adaptiveconcat = nn.Sequential(AdaptiveConcatPool2d(),
                                                )
        
        # Conv branch
        self.feat_conv = nn.Sequential(nn.Conv2d(nc, self.f_conv_out, (self.num_tiles,1), stride=(self.num_tiles,1)),
                                       Mish(),
                                       nn.Dropout(0.7),
                                       nn.BatchNorm2d(self.f_conv_out),
                                       AdaptiveConcatPool2d(), # this will duplicate self.f_conv_out
                                      )
        
        # create the classification head, where input size is 2*(nc+self.f_conv_out) and output n
        # input size is 2*(nc+self.f_conv_out) because we use AdaptiveConcatPool
        # and it's a layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`, thus 2*(nc+self.f_conv_out)
        # n=6 with isup_grades from 0 to 5
        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(2*(nc+self.f_conv_out),512),
            nn.BatchNorm1d(512), 
            Mish(),
            nn.Dropout(final_dropout),
            nn.Linear(512,n)
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
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        # x: bs x C x N*4 x 4
        x1 = self.feat_conv(x) # x1: bs x 2*f_conv_out
        x2 = self.feat_adaptiveconcat(x) #x2: bs x 2*ech
        #print(x1.shape, x2.shape)
        x = torch.cat([x1, x2], axis=1)
        x = self.head(x)
        return x