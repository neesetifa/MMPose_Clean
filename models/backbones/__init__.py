#from .cspnext import CSPNeXt
import copy
from .mobilenet_v2_config_version import MobileNetV2
from .shufflenet_v2 import ShuffleNetV2

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    backbone_class = eval(backbone_cfg .pop('type'))
    backbone = backbone_class(**backbone_cfg)
    return backbone

__all__ = ['MobileNetV2', 'ShuffleNetV2', 'build_backbone'] # 'CSPNeXt'
