import copy
# from .cspnext_pafpn import CSPNeXtPAFPN
# from .fmap_proc_neck import FeatureMapProcessor
# from .fpn import FPN
from .gap_neck import GlobalAveragePooling
# from .posewarper_neck import PoseWarperNeck
# from .yolox_pafpn import YOLOXPAFPN

def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    neck_class = eval(neck_cfg.pop('type'))
    neck = neck_class(**neck_cfg)
    return neck

__all__ = ['GlobalAveragePooling', 'build_neck']
