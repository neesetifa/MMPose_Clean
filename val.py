import os
import argparse
import pdb
import torch
from tqdm import tqdm

from models.pose_estimator import build_pose_estimator
from datasets import build_dataloader
from evaluation import build_metric
from utils.general import parse_config_file

def evaluate(model, dataloader, metric):
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}')

    model.eval()
    for idx, data_batch in pbar:
        with torch.no_grad():
            if type(model) in [torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel]:
                data = model.module.data_preprocessor(data = data_batch, training = False)
            else:
                data = model.data_preprocessor(data = data_batch, training = False)
            outputs = model(**data, mode = 'predict')
            metric.process(data_batch = data_batch, data_samples = outputs)

            pbar.set_description('Evaluating on validation dataset')

    results = metric.evaluate(len(dataloader.dataset)) # 每次执行完evaluate后会自动清空结果
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='model config py file')
    parser.add_argument('--pretrained_weight', type=str, default=None, help='model weight file')
    parser.add_argument('--checkpoint', type=str, default=None, help='ckeckpoint or weight')
    
    return parser.parse_args()



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert (args.cfg_file is None and args.pretrained_weight is None)^(args.checkpoint is None),\
        'pass checkpoint or cfg_file with pretrained_weight'

    if args.checkpoint:
        assert os.path.isdir(args.checkpoint), 'checkpoint should be a directory'
        files = os.listdir(args.checkpoint)
        for f in files:
            if f.endswith('.py'):
                args.cfg_file = os.path.join(args.checkpoint, f)
                break
        args.pretrained_weight = os.path.join(args.checkpoint, 'best.pth')
        
    configs = parse_config_file(args.cfg_file)
    weight = torch.load(args.pretrained_weight, map_location='cpu')
    ww = weight['model'] if 'model' in weight.keys() else weight

    # initialize
    model = build_pose_estimator(configs['model'])
    model.load_state_dict(ww)
    model = model.to(device)

    val_loader, val_dataset = build_dataloader(configs['val_dataloader'])

    evaluate_metric = build_metric(configs['val_evaluator'])
    evaluate_metric.dataset_meta = val_dataset.metainfo

    # evaluate
    results = evaluate(model, val_loader, evaluate_metric)
    results_str = ', '.join([f'{k}: {v:.4f}' for k,v in results.items()])
    print(f'Evaluation result: {results_str}')

if __name__ == '__main__':
    args = parse_args()
    main(args)


"""
python val.py --checkpoint work_dirs/202402041739_hm_mb_075/
"""
