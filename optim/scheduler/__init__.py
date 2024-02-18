import copy
# yapf: disable
from .lr_scheduler import (ConstantLR, CosineAnnealingLR, CosineRestartLR,
                           ExponentialLR, LinearLR, MultiStepLR, OneCycleLR,
                           PolyLR, ReduceOnPlateauLR, StepLR)

# from .momentum_scheduler import (ConstantMomentum, CosineAnnealingMomentum,
#                                  CosineRestartMomentum, ExponentialMomentum,
#                                  LinearMomentum, MultiStepMomentum,
#                                  PolyMomentum, ReduceOnPlateauMomentum,
#                                  StepMomentum)

from .param_scheduler import (ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              CosineRestartParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, OneCycleParamScheduler,
                              PolyParamScheduler,
                              ReduceOnPlateauParamScheduler,
                              StepParamScheduler, _ParamScheduler)


def build_scheduler(scheduler_cfgs, optimizer, dataloader = None) -> list:
    # Allow multiple scheduler
    if not isinstance(scheduler_cfgs, (list, tuple)):
        scheduler_cfgs = [scheduler_cfgs]

    schedulers = []
    for scheduler_cfg in scheduler_cfgs:
        scheduler_cfg_copy = copy.deepcopy(scheduler_cfg)
        scheduler_class = eval(scheduler_cfg_copy.pop('type'))
        scheduler_cfg_copy.update(optimizer = optimizer)
        
        convert_to_iter = scheduler_cfg_copy.pop('convert_to_iter_based', False)
        if convert_to_iter:
            assert dataloader is not None and scheduler_cfg_copy.get('by_epoch', True),\
                'Only epoch-based scheduler can be converted to iter-based, and `epoch_length` should be set'
            scheduler_cfg_copy.update(epoch_length = len(dataloader))
            scheduler = scheduler_class.build_iter_from_epoch(**scheduler_cfg_copy)
        else:
            scheduler = scheduler_class(**scheduler_cfg_copy)
            
        schedulers.append(scheduler)

    return schedulers


# yapf: enable
__all__ = ['ConstantLR', 'CosineAnnealingLR', 'CosineRestartLR',
           'ExponentialLR', 'LinearLR', 'MultiStepLR', 'OneCycleLR',
           'PolyLR', 'ReduceOnPlateauLR', 'StepLR',
           'build_scheduler',
           ]
