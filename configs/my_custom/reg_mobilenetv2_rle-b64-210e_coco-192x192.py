_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=80, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-5,
))

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[50, ],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(type='RegressionLabel', input_size=(192, 192))

# model settings
# input image range [-1,1]
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[128, 128, 128], # [123.675, 116.28, 103.53]
        std=[128, 128, 128], # [58.395, 57.12, 57.375]
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=0.75,
        out_indices=(7, ),  # output from which stage of the backbone
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RLEHead',
        in_channels=1280,
        num_joints=17,
        loss=dict(type='RLELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
    ),
)

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='images/train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=f'{data_root}person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}annotations/person_keypoints_val2017.json',
    score_mode='bbox_rle')
test_evaluator = val_evaluator

randomness = dict(seed = 3407)

# QAT configure
qat_pretrained_weight = 'work_dirs/202402271226/last.pth'
quant_info = 'work_dirs/202402271226/last_quant_info.pth'
configure_list = [{'quant_types': ['weight', 'input', 'output'],
                   'quant_bits': {'weight': 8, 'input': 8, 'output': 8},
                   'op_names': ['backbone.conv1.conv',]},
                   
                  {'quant_types': ['weight', 'output'],
                   'quant_bits': {'weight': 8, 'output': 8},
                   'op_names': ['backbone.block_0.1.dw_conv', 'backbone.block_0.1.pw_conv',
                                'backbone.block_1.1.conv', 'backbone.block_1.1.dw_conv', 'backbone.block_1.1.pw_conv',
                                'backbone.block_1.2.conv', 'backbone.block_1.2.dw_conv', 'backbone.block_1.2.pw_conv',
                                'backbone.block_2.1.conv', 'backbone.block_2.1.dw_conv', 'backbone.block_2.1.pw_conv',
                                'backbone.block_2.2.conv', 'backbone.block_2.2.dw_conv', 'backbone.block_2.2.pw_conv',
                                'backbone.block_2.3.conv', 'backbone.block_2.3.dw_conv', 'backbone.block_2.3.pw_conv',
                                'backbone.block_3.1.conv', 'backbone.block_3.1.dw_conv', 'backbone.block_3.1.pw_conv',
                                'backbone.block_3.2.conv', 'backbone.block_3.2.dw_conv', 'backbone.block_3.2.pw_conv',
                                'backbone.block_3.3.conv', 'backbone.block_3.3.dw_conv', 'backbone.block_3.3.pw_conv',
                                'backbone.block_3.4.conv', 'backbone.block_3.4.dw_conv', 'backbone.block_3.4.pw_conv',
                                'backbone.block_4.1.conv', 'backbone.block_4.1.dw_conv', 'backbone.block_4.1.pw_conv',
                                'backbone.block_4.2.conv', 'backbone.block_4.2.dw_conv', 'backbone.block_4.2.pw_conv',
                                'backbone.block_4.3.conv', 'backbone.block_4.3.dw_conv', 'backbone.block_4.3.pw_conv',
                                'backbone.block_5.1.conv', 'backbone.block_5.1.dw_conv', 'backbone.block_5.1.pw_conv',
                                'backbone.block_5.2.conv', 'backbone.block_5.2.dw_conv', 'backbone.block_5.2.pw_conv',
                                'backbone.block_5.3.conv', 'backbone.block_5.3.dw_conv', 'backbone.block_5.3.pw_conv',
                                'backbone.block_6.1.conv', 'backbone.block_6.1.dw_conv', 'backbone.block_6.1.pw_conv',
                                'backbone.conv2.conv', 
                                ]},
                  
                  {'quant_types': ['output'],
                   'quant_bits': {'output': 8},
                   'op_names': ['backbone.block_1.2.add',
                                'backbone.block_2.2.add',
                                'backbone.block_2.3.add',
                                'backbone.block_3.2.add',
                                'backbone.block_3.3.add',
                                'backbone.block_3.4.add',
                                'backbone.block_4.2.add',
                                'backbone.block_4.3.add',
                                'backbone.block_5.2.add',
                                'backbone.block_5.3.add',
                                'neck.gap', # avg_pool
                                ]},
                  
                  {'quant_types': ['weight', 'output'],
                   'quant_bits': {'weight': 8, 'output': 8},
                   'op_names': ['head.fc_coord', 'head.fc_sigma']}]
