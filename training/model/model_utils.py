import torch
import fastai
from fastai.vision import *
from model.model_config import ModelConfig
from model import get_panda_model

def load_models_from_dir(model_dir, tile_list_input=True):
    # load config
    config = ModelConfig.fromDir(model_dir)
    
    n_folds = config.getMetaField('n_folds')
    if n_folds is None: n_folds = 4
    
    model_file_prefix = config.getMetaField('model_file_prefix')
    if model_file_prefix is None: model_file_prefix = ""

    model_name = config.getField('model_name')
    arch = config.getField('arch')
    model_n_out = config.getField('model_n_out')
    N = config.getField('N')

    # load models
    model_paths = [os.path.join(model_dir, f'{model_file_prefix}{i}.pth') for i in range(n_folds)]
    model_func = get_panda_model(model_name, arch, n=model_n_out, num_tiles=N, pretrained=False, is_train=False, tile_list_input = tile_list_input)

    models = []
    for model_path in model_paths:
        #assert os.path.isfile(model_path), f'Model not found {model_path}'
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)#,map_location=torch.device('cpu'))
            model = model_func()
            model.load_state_dict(state_dict)
            model.float()
            model.eval()
            model.cuda()
            models.append(model)
    
    return models

def load_weights(model_dir, fold):
    # load config
    config = ModelConfig.fromDir(model_dir)
    
    n_folds = config.getMetaField('n_folds')
    if n_folds is None: n_folds = 4
    
    model_file_prefix = config.getMetaField('model_file_prefix')
    if model_file_prefix is None: model_file_prefix = ""

    # load models
    model_path = os.path.join(model_dir, f'{model_file_prefix}{fold}.pth')
    assert os.path.isfile(model_path), f'Model not found {model_path}'

    state_dict = torch.load(model_path)
    return state_dict
