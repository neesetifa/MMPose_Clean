# Copyright (c) OpenMMLab. All rights reserved.
from .integral_regression_label import IntegralRegressionLabel
from .megvii_heatmap import MegviiHeatmap
from .msra_heatmap import MSRAHeatmap
from .regression_label import RegressionLabel
from .simcc_label import SimCCLabel
from .udp_heatmap import UDPHeatmap
from .simcc_regression_label import SimCCRegLabel

def build_codec(cfg):
    codec_class = eval(cfg.pop('type'))
    codec = codec_class(**cfg)
    return codec


__all__ = [
    'MSRAHeatmap', 'MegviiHeatmap', 'UDPHeatmap', 'RegressionLabel',
    'SimCCLabel', 'IntegralRegressionLabel', 'SimCCRegLabel',
    'build_codec'
]
