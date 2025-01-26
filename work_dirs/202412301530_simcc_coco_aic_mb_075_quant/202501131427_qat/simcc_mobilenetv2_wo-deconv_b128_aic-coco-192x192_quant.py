_base_ = ['../_base_/default_runtime.py']

# runtime
max_epochs = 80
stage2_num_epochs = 30
base_lr = 1e-4

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=3407)

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0),
#     # 参数说明
#     # https://github.com/open-mmlab/mmengine/blob/2c4516c62294964065d058d98799402f50afdef6/docs/zh_cn/tutorials/optim_wrapper.md?plain=1#L295
#     paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
# )

optim_wrapper = dict(optimizer=dict(type='AdamW',
                                    lr=base_lr,
                                    weight_decay=0.0,
                                    ),
                     clip_grad=dict(max_norm=10.0)  # 新加的
                     )


# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 192),
    sigma=(4.9, 4.9),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
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
        out_indices=(7, ),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='pretrained_weight/2_udp_heatmap/mobilenetv2_0.75x_udp_coco_aic_pretrained_backbone_192x192.pth',
        )
    ),
    head=dict(
        type='SimCCHead',
        in_channels=1280,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        deconv_out_channels=None,
        loss=dict(type='KLDiscretLoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))


# pipelines
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline = [
    # stage-2 of original training
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# base dataset settings
data_mode = 'topdown'
data_root = 'data/'

# train datasets
dataset_coco = dict(
    type='RepeatDataset',
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='coco/images/train2017/'),
        pipeline=[],
    ),
    times=3)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='aic/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[
                (0, 6),
                (1, 8),
                (2, 10),
                (3, 5),
                (4, 7),
                (5, 9),
                (6, 12),
                (7, 14),
                (8, 16),
                (9, 11),
                (10, 13),
                (11, 15),
            ])
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[dataset_coco, dataset_aic],
        pipeline=train_pipeline,
        test_mode=False,
    ))
val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        bbox_file=f'{data_root}/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='coco/images/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader


# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator


# QAT configure
qat_mode = 'qat_fixed'  # lsq+, qat_fixed
qat_pretrained_weight = 'work_dirs/202501101609_lsq/last.pth'
quant_info = 'work_dirs/202501101609_lsq/last_quant_info.pth'
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
                                ]},
                  
                  {'quant_types': ['weight', 'output'],
                   'quant_bits': {'weight': 8, 'output': 8},
                   'op_names': ['head.final_layer', 'head.mlp_head_x', 'head.mlp_head_y']}]
