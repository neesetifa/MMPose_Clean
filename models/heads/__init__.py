import copy
from .base_head import BaseHead
from .coord_cls_heads import SimCCHead
from .heatmap_heads import HeatmapHead, ViPNASHead
from .hybrid_heads import SimCCRLEHead
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               RegressionHead, RLEHead)


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    head_class = eval(head_cfg.pop('type'))
    head = head_class(**head_cfg)
    return head

__all__ = ['BaseHead',
           'SimCCHead',
           'HeatmapHead', 'ViPNASHead',
           'RegressionHead', 'IntegralRegressionHead', 'RLEHead', 'DSNTHead',
           'build_head'
           ]
