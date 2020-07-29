
from functools import partial
import torch
from model.mxresnet import get_mxresnet_func

"""
When adding models, make sure they follow this interface:

MyModel(arch=arch_string_identifier, n=number_of_outputs, final_dropout=final_dropout, pre=is_pretrained)

arch: e.g. resnext50_32x4d_ssl
n (int): n model outputs
final_dropout (float): final dropout
pre (bool): pretrained


Add new models to model_func()'s if-elif-statement 

"""

__all__ = [
    'get_panda_model',
]

resnest_architectures = ['resnest50', 'resnest101', 'resnest200', 'resnest269']
mxresnet_architectures = ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']

def get_encoder(arch='resnext50_32x4d_ssl', pre=True, self_attention=1, self_attention_symmetry=0):
    # Load the encoder model with weights from torch.hub
    if arch in resnest_architectures:
        return torch.hub.load('zhanghang1989/ResNeSt', arch, pretrained=pre)
    elif arch in mxresnet_architectures:
        mxfunc = get_mxresnet_func(arch)
        return mxfunc(c_out=6, sa=self_attention,sym=self_attention_symmetry)
    else:
        return torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)

def get_panda_model(model_name, arch='resnext50_32x4d_ssl', n=6, final_dropout=0.5,
                    pretrained=True, num_tiles=12, f_conv_out=128,
                    is_train=True, tile_list_input=True, self_attention=1, self_attention_symmetry=0,**kwargs):

    enc_func = partial(get_encoder, arch, pretrained, self_attention=self_attention, self_attention_symmetry=self_attention_symmetry)

    def model_func():
        # Add new models as elif statements
        if model_name == "iafoss":
            from .iafoss_model import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input)
            
        elif model_name == "conv_branch":
            from .conv_branch_model import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input, num_tiles=num_tiles, f_conv_out=f_conv_out)

        elif model_name == "gem_concat_pooling":
            from .gem_concat_pooling_model import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input)

        elif model_name == "multihead_tilecat":
            from .multihead_tilecat import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input, num_tiles=num_tiles, is_train=is_train)
        
        elif model_name == "multihead_tilecat_attention":
            from .multihead_tilecat_attention import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input, num_tiles=num_tiles, is_train=is_train)
        
        
        elif model_name == "iafoss_regr":
            from .iafoss_regr_model import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input)
        
        elif model_name == "conv_branch_regr":
            from .conv_branch_regr_model import Model
            return Model(enc_func=enc_func, n=n, final_dropout=final_dropout, tile_list_input = tile_list_input, num_tiles=num_tiles, f_conv_out=f_conv_out)
        else:
            print("{0} base model not found".format(model_name))


    return model_func