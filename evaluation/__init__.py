# Copyright (c) OpenMMLab. All rights reserved.
import copy
#from .evaluators import *  # noqa: F401,F403
from .functional import *  # noqa: F401,F403
from .metrics import *  # noqa: F401,F403

def build_metric(cfg, outfile_prefix = None):
    metric_cfg = copy.deepcopy(cfg)
    metric_class = eval(metric_cfg.pop('type'))
    # outfile_prefix piority:
    # 1. config file 2. training save dir 3. None
    if 'outfile_prefix' not in metric_cfg:
        metric_cfg.update(outfile_prefix = outfile_prefix)
    metric = metric_class(**metric_cfg)
    return metric


__all__ = ['build_metric',
           'CocoMetric',
           'keypoint_pck_accuracy', 'keypoint_auc', 'keypoint_nme', 'keypoint_epe',
    'pose_pck_accuracy', 'multilabel_classification_accuracy',
    'simcc_pck_accuracy', 'nms', 'oks_nms', 'soft_oks_nms', 'keypoint_mpjpe',
    'nms_torch', 'transform_ann', 'transform_sigmas', 'transform_pred']
