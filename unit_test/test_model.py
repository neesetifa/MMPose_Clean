# 工作路径切换到上一层
import os
import sys
ori_path = os.getcwd()
new_path = os.path.dirname(ori_path)
os.chdir(new_path)
sys.path.append(new_path)

import pdb
import torch

from models.pose_estimator import build_pose_estimator


if __name__ == '__main__':
    # configs
    #codec = dict(type='SimCCLabel', input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0) # [w,h]
    #codec = dict(type='SimCCRegLabel', input_size=(192, 192), sigma=6.0, simcc_split_ratio=2.0)
    codec = dict(type='RegressionLabel', input_size=(192, 192))
    
    cfg = dict(type='TopdownPoseEstimator',
               data_preprocessor=dict(type='PoseDataPreprocessor',
                                      mean=[123.675, 116.28, 103.53],
                                      std=[58.395, 57.12, 57.375],
                                      bgr_to_rgb=True),
               backbone=dict(type='MobileNetV2', widen_factor=0.5),
               # head=dict(type='SimCCHead',
               #           in_channels=1280,
               #           out_channels=17,
               #           input_size=codec['input_size'],
               #           in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
               #           simcc_split_ratio=codec['simcc_split_ratio'],
               #           deconv_out_channels=None,
               #           loss=dict(type='KLDiscretLoss', use_target_weight=True),
               #           decoder=codec),
               # head=dict(type='SimCCRLEHead',
               #           in_channels=1280,
               #           out_channels=17,
               #           input_size=codec['input_size'],
               #           in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
               #           num_joints=17,
               #           simcc_split_ratio=codec['simcc_split_ratio'],
               #           deconv_out_channels=None,
               #           loss=dict(type='CombinedLoss',
               #                     losses = dict(simcc_loss = dict(type='KLDiscretLoss',
               #                                                     use_target_weight=True),
               #                                   rle_loss = dict(type='RLELoss',
               #                                                   use_target_weight=True),
               #                                   ),
               #                     losses_weight = dict(simcc_loss_weight = 1.0,
               #                                          rle_loss_weight = 0.01,
               #                                          ),
               #                     ),
               #           decoder=codec),
               neck=dict(type='GlobalAveragePooling'),
               head=dict(type='RLEHead',
                         in_channels=1280,
                         num_joints=17,
                         loss=dict(type='RLELoss', use_target_weight=True),
                         decoder=codec),
               test_cfg=dict(flip_test=True, ),
               )

    # build code
    model = build_pose_estimator(cfg)
    from torchsummaryX import summary
    model.eval()
    dummy_input = torch.zeros((1, 3, codec['input_size'][1], codec['input_size'][0])) # [C,H,W]
    summary(model, dummy_input) 
    pdb.set_trace()
