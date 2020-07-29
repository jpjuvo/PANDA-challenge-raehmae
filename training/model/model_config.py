import json
import os
import numpy as np

class ModelConfig:
    """
    Contains all necessary information to use the model for inference. May also include training metadata.
    Model directory should contain config.json and this class can be directly initialized from a model dir with fromDir.  
    """

    def __init__(self, model_name, arch, model_n_out, sz, N, mean, std, meta={}):
        self.config_dict = {'model_name':model_name,
                            'arch':arch,
                            'model_n_out':model_n_out,
                            'sz':sz,
                            'N':N,
                            'mean':list(mean.astype(str)),
                            'std':list(std.astype(str)),
                            'meta':meta}

    @classmethod
    def fromDir(cls, dir_path):
        config_path = os.path.join(dir_path,'config.json')
        with open(config_path) as json_file:
            data = json.load(json_file)
            model_name = data['model_name']
            arch = data['arch']
            model_n_out = data['model_n_out']
            sz = data['sz']
            N = data['N']
            mean = np.array(data['mean'])
            std = np.array(data['std'])
            meta = data['meta']
        return cls(model_name, arch, model_n_out, sz, N, mean, std, meta)

    def toDir(self, dir_path):
        config_path = os.path.join(dir_path,'config.json')
        with open(config_path, 'w') as outfile:
            json.dump(self.config_dict, outfile, indent=4)

    def getField(self, field):
        if field in self.config_dict.keys():
            return self.config_dict[field]
        return None

    def getMetaField(self, field):
        if field in self.config_dict['meta'].keys():
            return self.config_dict['meta'][field]
        return None
    
