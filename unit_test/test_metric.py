# 工作路径切换到上一层
import os
import sys
ori_path = os.getcwd()
new_path = os.path.dirname(ori_path)
os.chdir(new_path)
sys.path.append(new_path)

import pdb
import torch

from evaluation import build_metric

if __name__ == '__main__':
    # configs
    data_root = 'data/coco/'
    val_evaluator = dict(type='CocoMetric',
                         ann_file=f'{data_root}annotations/person_keypoints_val2017.json',
                         score_mode='bbox_rle')

    evaluator = build_metric(val_evaluator)
    pdb.set_trace()
