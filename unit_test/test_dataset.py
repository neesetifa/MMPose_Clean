# 工作路径切换到上一层
import os
import sys
ori_path = os.getcwd()
new_path = os.path.dirname(ori_path)
os.chdir(new_path)
sys.path.append(new_path)

import pdb
import cv2
import numpy as np
import torch

from datasets import build_dataloader

if __name__ == '__main__':
    # configs
    codec = dict(type='SimCCLabel', input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0)
    # codec = dict(type='RegressionLabel', input_size=(192, 256))
    # codec = dict(type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

    train_pipeline = [dict(type='LoadImage'),
                      dict(type='GetBBoxCenterScale'),
                      dict(type='RandomFlip', direction='horizontal'),
                      dict(type='RandomHalfBody'),
                      dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4]),
                      dict(type='TopdownAffine', input_size=codec['input_size']),
                      dict(type='Albumentation',
                           transforms=[dict(type='Blur', p=0.1),
                                       dict(type='MedianBlur', p=0.1),
                                       dict(type='CoarseDropout',  # ❓ 遮盖的是不是有点多
                                            max_holes=1,
                                            max_height=0.4,
                                            max_width=0.4,
                                            min_holes=1,
                                            min_height=0.2,
                                            min_width=0.2,
                                            p=1.0),
                                       ]),
                      dict(type='GenerateTarget', encoder=codec),
                      dict(type='PackPoseInputs', pack_transformed=True) # pack_transformed for debug
                      ]

    train_pipeline_stage2 = [dict(type='LoadImage'),
                             dict(type='GetBBoxCenterScale'),
                             dict(type='TopdownAffine', input_size=codec['input_size']),
                             dict(type='YOLOXHSVRandomAug'),
                             dict(type='GenerateTarget', encoder=codec),
                             dict(type='PackPoseInputs', pack_transformed=True) # pack_transformed for debug
                             ]

    val_pipeline = [dict(type='LoadImage'),
                    dict(type='GetBBoxCenterScale'),
                    dict(type='TopdownAffine', input_size=codec['input_size']),
                    dict(type='PackPoseInputs') # no pack_transformed cuz no label in val
                    ]

    # Choose one of the dataset
    # --- Test-1 Single Dataset
    # dataset_coco = dict(type='CocoDataset',
    #                     data_root='data/coco/',
    #                     data_mode='topdown',
    #                     ann_file='annotations/person_keypoints_train2017.json',
    #                     data_prefix=dict(img='images/train2017/'),
    #                     pipeline=train_pipeline,
    #                     pipeline_stage2=train_pipeline_stage2,
    #                     )

    # train_dataloader_cfg = dict(batch_size=64,
    #                             num_workers=1, # 2
    #                             persistent_workers=True,
    #                             sampler=dict(type='DefaultSampler', shuffle=False),
    #                             dataset = dataset_coco,
    #                             )

    # --- Test-2 Multiple Datasets
    dataset_coco = dict(type='CocoDataset',
                        data_root='data/coco/',
                        data_mode='topdown',
                        ann_file='annotations/person_keypoints_train2017.json',
                        data_prefix=dict(img='images/train2017/'),
                        pipeline=[],
                        ) # total 149813
    
    dataset_aic = dict(type='AicDataset',
                       data_root='data/aic',
                       data_mode='topdown',
                       ann_file='annotations/aic_train.json',
                       data_prefix=dict(img='ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'),
                       pipeline=[dict(type='KeypointConverter',
                                      num_keypoints=17,
                                      mapping=[(0, 6),
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
                       ) # total 378352

    train_dataloader_cfg = dict(batch_size=64,
                                num_workers=1, # 2
                                persistent_workers=True,
                                sampler=dict(type='DefaultSampler', shuffle=False),
                                dataset = dict(type='CombinedDataset',
                                               metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
                                               datasets=[dataset_aic, dataset_coco],
                                               pipeline=train_pipeline,
                                               pipeline_stage2=train_pipeline_stage2,
                                               test_mode=False,
                                               )
                                )
    

    val_dataloader_cfg = dict(batch_size=32,
                              num_workers=2,
                              persistent_workers=True,
                              drop_last=False,
                              sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
                              dataset=dict(type='CocoDataset',
                                           data_root='data/coco/',
                                           data_mode='topdown',
                                           ann_file='annotations/person_keypoints_val2017.json',
                                           bbox_file='data/coco/person_detection_results/'
                                           'COCO_val2017_detections_AP_H_56_person.json',
                                           data_prefix=dict(img='images/val2017/'),
                                           test_mode=True,
                                           pipeline=val_pipeline,
                                           )
                              )
    
    # build code
    train_dataloader, train_dataset = build_dataloader(train_dataloader_cfg)
    # val_dataloader, val_dataset = build_dataloader(val_dataloader_cfg)

    # test code
    # Training pipeline - stage 1
    for idx, data in enumerate(train_dataloader):
        images = data['inputs'] # [batch, 3, 256, 192], in uint8, BGR format
        meta_infos = data['data_samples'] # [batch, ...]
        print('Train Batch size:', len(images))

        # get info of first image in batch data
        image = images[0]
        meta_info = meta_infos[0]
        original_keypoints = meta_info.gt_instances.keypoints[0] # [17,2]
        transformed_keypoints = meta_info.gt_instances.transformed_keypoints[0] # [17,2]
        keypoints_visible = meta_info.gt_instances.keypoints_visible[0] # [17]
        
        if hasattr(meta_info, 'gt_fields'):
            # Heatmap
            training_label = meta_info.gt_fields.heatmaps  # [17, heatmap_h=64, heatmap_w=48]
        elif hasattr(meta_info.gt_instance_labels, 'keypoint_x_labels'):
            # Classification
            training_label = [meta_info.gt_instance_labels.keypoint_x_labels, # [17, input_w*2=384]
                              meta_info.gt_instance_labels.keypoint_y_labels] # [17, input_h*2=512]
        elif hasattr(meta_info.gt_instance_labels, 'keypoint_labels'):
            # Regression
            training_label = meta_info.gt_instance_labels.keypoint_labels # [17,2]
        else:
            print('Invalid label style')
            pdb.set_trace()
        
        # test input image
        im = image.numpy()
        im = np.transpose(im,(1,2,0))
        im = np.ascontiguousarray(im)
        for i in range(len(keypoints_visible)):
            if keypoints_visible[i]:
                x, y = transformed_keypoints[i]
                cv2.circle(im, (int(x), int(y)), 2, (153, 0, 255), 2)
        cv2.imwrite(os.path.join(ori_path, f'{idx}_t.jpg'), im)
        # print(training_label)
        
        pdb.set_trace()
        if idx > 10:
            break

    # # Validation pipeline
    # for idx, data in enumerate(val_dataloader):
    #     images = data['inputs'] # [batch, 3, 256, 192], in uint8, BGR format
    #     labels = data['data_samples'] # [batch, ...] no useful info in it
    #     print('Val Batch size:', len(images))
        
    #     # test input image
    #     im = images[0].numpy()
    #     im = np.transpose(im,(1,2,0))
    #     cv2.imwrite(os.path.join(ori_path, f'{idx}_v.jpg'), im)
        
    #     pdb.set_trace()
    #     if idx > 1:
    #         break


    # Training pipeline - stage 2
    # The dataset pipeline cannot be updated when persistent_workers
    # is True, so we need to force the dataloader's multi-process
    # restart. This is a very hacky approach.
    train_dataloader.dataset.switch_pipeline()
    if hasattr(train_dataloader, 'persistent_workers') and train_dataloader.persistent_workers is True:
        train_dataloader._DataLoader__initialized = False
        train_dataloader._iterator = None
    # train_dataloader._DataLoader__initialized = True
        
    for idx, data in enumerate(train_dataloader):
        images = data['inputs'] # [batch, 3, 256, 192], in uint8, BGR format
        meta_infos = data['data_samples'] # [batch, ...]
        print('Train Batch size:', len(images))

        # get info of first image in batch data
        image = images[0]
        meta_info = meta_infos[0]
        original_keypoints = meta_info.gt_instances.keypoints[0] # [17,2]
        transformed_keypoints = meta_info.gt_instances.transformed_keypoints[0] # [17,2]
        keypoints_visible = meta_info.gt_instances.keypoints_visible[0] # [17]
        
        # test input image
        im = image.numpy()
        im = np.transpose(im,(1,2,0))
        im = np.ascontiguousarray(im)
        for i in range(len(keypoints_visible)):
            if keypoints_visible[i]:
                x, y = transformed_keypoints[i]
                cv2.circle(im, (int(x), int(y)), 2, (153, 0, 255), 2)
        cv2.imwrite(os.path.join(ori_path, f'{idx}_t2.jpg'), im)
        
        pdb.set_trace()
        if idx > 10:
            break
