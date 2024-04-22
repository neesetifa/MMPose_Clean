import pdb
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import HistogramObserver

from models.pose_estimator import build_pose_estimator
from datasets import build_dataloader
from utils.general import parse_config_file
from utils.env import init_seeds

from nni.algorithms.compression.pytorch.quantization import PTQ_Quantizer

torch.multiprocessing.set_sharing_strategy('file_system')


def ptq(model, train_loader, stop = 100):
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}')

    model.eval()

    for idx, data_batch in pbar:  # batch -------------------------------------------------------------
        with torch.no_grad():
            if type(model) in [nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]:
                data = model.module.data_preprocessor(data = data_batch, training = True)
            else:
                data = model.data_preprocessor(data = data_batch, training = True)
            _ = model(**data, mode = 'tensor') # tensor mode is enough for PTQ
        
            pbar.set_description('Running PTQ on training dataset')
            
        if idx > stop:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='model config py file')
    parser.add_argument('--pretrained_weight', type=str, default=None, help='model weight file')
    parser.add_argument('--save_dir', default='', help='save to project/name')
    
    return parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.save_dir == '':
        args.save_dir = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = os.path.join('work_dirs', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    configs = parse_config_file(args.cfg_file)
    randomness_cfg = configs.get('randomness', {'seed': 3407})
    init_seeds(**randomness_cfg, rank=0)

    # initialize
    weight = torch.load(args.pretrained_weight, map_location='cpu')
    ww = weight['model'] if 'model' in weight.keys() else weight

    model = build_pose_estimator(configs['model'])
    model.load_state_dict(ww)

    w, h = configs["codec"]["input_size"]
    INPUT_SIZE = [1, 3, h, w] # N,C,H,W
    dummy_inputs = torch.randn(*INPUT_SIZE)
    configure_list = configs['configure_list']
    model.eval()
    quantizer = PTQ_Quantizer(model, configure_list, dummy_input=dummy_inputs)
    quantizer.compress()

    model = model.to(device)
    train_loader, train_dataset = build_dataloader(configs['train_dataloader'])

    # run inference
    ptq(model, train_loader)
    quantizer.save_model(model_path=os.path.join(args.save_dir, 'last.pth'))
    print('====== PTQ finished ========')
    print(f'Quant info saved in {args.save_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)


"""
** Single GPU PTQ
python ptq.py --cfg_file configs/my_custom/reg_mobilenetv2_rle_b256_aic-coco-192x192_quant.py --pretrained_weight work_dirs/202404121425_rle_coco_aic_mb_075/best.pth
"""
