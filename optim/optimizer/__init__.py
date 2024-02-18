import copy
# from .amp_optimizer_wrapper import AmpOptimWrapper
# from .apex_optimizer_wrapper import ApexOptimWrapper
# from .default_constructor import DefaultOptimWrapperConstructor
from .optimizer_wrapper import OptimWrapper
# from .optimizer_wrapper_dict import OptimWrapperDict
# from .zero_optimizer import ZeroRedundancyOptimizer

from torch.optim import * 
def build_optimizer_wrapper(cfg, model):
    cfg_copy = copy.deepcopy(cfg)
    wrapper_class = eval(cfg_copy.pop('type', 'OptimWrapper'))
    optimizer_cfg = cfg_copy.pop('optimizer')
    optimizer_class = optimizer_cfg.pop('type')
    optimizer_cfg.update(params = model.parameters())
    optimizer = eval(optimizer_class)(**optimizer_cfg)
    cfg.update(optimizer = optimizer)
    assert 'paramwise_cfg' not in cfg, \
        'Optim Constructor not support yet, but got paramwise_cfg in config'
    optimizer_wrapper = wrapper_class(optimizer = optimizer, **cfg_copy)
    return optimizer_wrapper


__all__ = ['OptimWrapper', 'build_optimizer_wrapper'
           ]
