import copy
from .data_preprocessor import BaseDataPreprocessor, PoseDataPreprocessor

def build_preprocessor(cfg):
    preprocessor_cfg = copy.deepcopy(cfg)
    preprocessor_class = eval(preprocessor_cfg.pop('type'))
    preprocessor = preprocessor_class(**preprocessor_cfg)
    return preprocessor

__all__ = ['BaseDataPreprocessor', 'PoseDataPreprocessor', 'build_preprocessor']
