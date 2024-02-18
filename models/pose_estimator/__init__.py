import copy
from .topdown import TopdownPoseEstimator

def build_pose_estimator(cfg):
    model_cfg = copy.deepcopy(cfg)
    model_class = eval(model_cfg.pop('type'))
    model = model_class(**model_cfg)
    return model

__all__ = ['TopdownPoseEstimator', 'build_pose_estimator']
