import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from train.mish_activation import *
from torch.nn.parameter import Parameter

def gem(x, p=3, eps=1e-6):
    p.clamp(1.,1000.)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=1, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class AdaptiveGeMConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.gem = GeM()
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): 
        return torch.cat([self.mp(x), self.gem(x)], 1)

class Model(nn.Module):
    def __init__(self, enc_func, n=6,final_dropout=0.5, tile_list_input=True, pre=True):
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
            AdaptiveGeMConcatPool2d(),
            Flatten(),
            nn.Linear(2*nc,512),
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
        x = self.head(x)
        return x