import pdb
import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
# Refer to following issues:
# https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


from models.pose_estimator import build_pose_estimator
from datasets import build_dataloader
from evaluation import build_metric
from optim import build_optimizer_wrapper, build_scheduler
from utils.logger import LOGGER, TQDM_BAR_FORMAT
from utils.env import select_device, init_seeds
from utils.dist import smart_DDP
from utils.general import parse_config_file, save_config_file, print_args, CombinedMeter

from val import evaluate

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer_fixed, LSQplus_Quantizer, QAT_Quantizer_old

# DEVICE info for current environment
# Initialize before running any main function
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


"""
before_run:
(VERY_HIGH   ) RuntimeInfoHook                    
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
before_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(NORMAL      ) DistSamplerSeedHook                
 -------------------- 
before_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_train_iter:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) SyncBuffersHook                    
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_val_epoch:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) SyncBuffersHook                    
 -------------------- 
before_val_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_val_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) PoseVisualizationHook              
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_val_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(BELOW_NORMAL) LoggerHook                         
(LOW         ) ParamSchedulerHook                 
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
after_val:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_train:
(VERY_HIGH   ) RuntimeInfoHook                    
(VERY_LOW    ) CheckpointHook                     
 -------------------- 
before_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
before_test_epoch:
(NORMAL      ) IterTimerHook                      
 -------------------- 
before_test_iter:
(NORMAL      ) IterTimerHook                      
 -------------------- 
after_test_iter:
(NORMAL      ) IterTimerHook                      
(NORMAL      ) PoseVisualizationHook              
(NORMAL      ) BadCaseAnalysisHook                
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test_epoch:
(VERY_HIGH   ) RuntimeInfoHook                    
(NORMAL      ) IterTimerHook                      
(NORMAL      ) BadCaseAnalysisHook                
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
after_test:
(VERY_HIGH   ) RuntimeInfoHook                    
 -------------------- 
after_run:
(BELOW_NORMAL) LoggerHook                         
 -------------------- 
"""


def train(args, configs, device):
    # Initialize and settings ✅
    cuda = device.type != 'cpu'
    randomness_cfg = configs.get('randomness', {'seed': 3407})
    init_seeds(**randomness_cfg, rank=RANK)
    start_epoch = 0
    best_mAP = 0.
    best_epoch = -1
    max_epochs = configs['train_cfg']['max_epochs']
    switch_epoch = (max_epochs-1) - configs.get('stage2_num_epochs', -10) # if no switch epoch, just make it larger than max_epoch
    
    if RANK in {-1, 0}:
        save_config_file(args.cfg_file, args.save_dir)
        
    train_batch_size = configs['train_dataloader']['batch_size']//WORLD_SIZE  # In config, training batch size is for all GPUs,
    configs['train_dataloader'].update(batch_size = train_batch_size)         # No need to update val batch, we only do eval on single GPU

    if args.resume_ckpt:
        assert not args.quant, 'QAT mode doesn\'t support resume checkpoint yet'
        ckpt = torch.load(args.checkpoint, map_location='cpu') # load checkpoint to CPU to avoid CUDA memory leak
        LOGGER.info(f'Load checkpoint from {args.checkpoint}')

    # Train loader ✅
    train_loader, train_dataset = build_dataloader(configs['train_dataloader'])
    
    # Val loader(only on Process 0) ✅
    if RANK in {-1, 0}:
        val_loader, val_dataset = build_dataloader(configs['val_dataloader'])
    
    # Model ✅
    model = build_pose_estimator(configs['model'])
    # for name, module in model.named_modules():
    #     if len(list(module.named_children())) > 0:
    #            continue
    #     print(name, type(module).__name__)
    # pdb.set_trace()
    
    if args.resume_ckpt:
        model.load_state_dict(ckpt['model'])
    else:
       model.init_weights() # currently only init backbone
    model = model.to(device)

    # Optimizer ✅
    optimizer_wrapper = build_optimizer_wrapper(configs['optim_wrapper'], model)

    # QAT ☑️
    # TODO - After new QAT, this part should be moved right after model definition
    if args.quant:
        model = model.to('cpu')
        assert configs['qat_pretrained_weight'] is not None, \
            'You must provide pretrained weight if you enable QAT, but found qat_pretrained_weight is None in configure file'
        ww = torch.load(configs['qat_pretrained_weight'], map_location='cpu')
        model.load_state_dict(ww['model'] if 'model' in ww.keys() else ww)
        LOGGER.info(f'Load pretrained weight from: {configs["qat_pretrained_weight"]}')

        w, h = configs["codec"]["input_size"]
        INPUT_SIZE = [1, 3, h, w] # N,C,H,W
        dummy_inputs = torch.randn(*INPUT_SIZE)

        configure_list = configs['configure_list']
        if configs['quant_info'] is None:
            quant_info = {}
        else:
            quant_info = torch.load(configs['quant_info'], map_location='cpu')
            LOGGER.info(f'Load quant_info from: {configs["quant_info"]}')

        model.eval() 
        optimizer = optimizer_wrapper.optimizer
        # model.forward()默认行为是接受处理好的input, 然后运行backbone->neck->head
        # 所以不需要做特殊处理即可送给Quantizer
        if configs['qat_mode'] == 'lsq+':
            quantizer_module = LSQplus_Quantizer
        elif configs['qat_mode'] == 'qat_fixed':
            quantizer_module = QAT_Quantizer_fixed
        # elif configs['qat_mode'] == 'qat_old':
        #     quantizer_module = QAT_Quantizer_old
        else:
            raise ValueError(f'qat mode must be lsq+ or qat_fixed, but got {configs["qat_mode"]}')
        # 被quantizer绑定后, QAT会多出一套学习率(new_weight), LSQ+会多出两套学习率(new_weight, 所有的scale)
        # 这个会在tqdm bar里显示出来
        quantizer = quantizer_module(model, configure_list, optimizer, dummy_inputs, quant_info)
        quantizer.compress()
        print('Quantized module wrapped successfully =========')
        # pdb.set_trace()
        model = model.to(device)
    
    # Schedulers ✅
    schedulers = build_scheduler(configs['param_scheduler'], optimizer_wrapper, train_loader) # return type: List

    # Resume ✅
    if args.resume_ckpt:
        assert not args.quant, 'QAT mode doesn\'t support Resume yet'
        start_epoch = ckpt['epoch']+1
        best_mAP, best_epoch = ckpt['best_mAP']
        optimizer_wrapper.load_state_dict(ckpt['optimizer_wrapper'])
        for sch, state_dict in zip(schedulers, ckpt['schedulers']):
            sch.load_state_dict(state_dict)
        LOGGER.info('Successfully resume optimizer and schedulers from checkpoint')

    # DP mode ✅
    if not args.quant and cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm ✅
    if args.sync_bn and cuda and RANK != -1:
        # During QAT, running mean and bias are fixed, so no need for sync
        assert not args.quant, 'QAT mode doesn\'t support SyncBatchNorm'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Evaluation metric ✅
    evaluate_metric = build_metric(configs['val_evaluator'], args.save_dir+'/val_results')
    evaluate_metric.dataset_meta = val_dataset.metainfo

    # DDP mode ✅
    # QAT doesn't support DDP yet
    if cuda and RANK != -1:
        assert not args.quant, 'QAT mode doesn\'t support DDP yet'
        model = smart_DDP(model)

    # 💎💎 Before train 💎💎
    # -- Initialize ✅
    training_start_time = time.time()
    optimizer_wrapper.initialize_count_status(init_counts = 0,
                                              max_counts = max_epochs*len(train_loader)
                                              )
    restart_dataloader = False
    METERS = CombinedMeter()

    LOGGER.info(f'Model input image sizes(w,h): {configs["codec"]["input_size"]}\n'
                f'Total num of training: {len(train_dataset)}\n'
                f'Total num of validation {len(val_dataset)}\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers on training\n'
                f'All training results saved to ==> {args.save_dir}\n'
                f'Start epoch {start_epoch}\n'
                f'Starting training for {max_epochs-start_epoch} epochs...')

    # 💎💎 Start train 💎💎
    for epoch in range(start_epoch, max_epochs):  # epoch ------------------------------------------------------------------
        # 🔶🔶 Before train epoch 🔶🔶
        # # -- timer(NORMAL) ☑️
        # epoch_time = time.time()

        # -- Sampler seed ✅
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        
        # -- Switch dataset pipeline if it has a second pipeline ✅
        # The dataset pipeline cannot be updated when persistent_workers
        # is True, so we need to force the dataloader's multi-process
        # restart. This is a very hacky approach.
        if train_loader.dataset.pipeline_stage2 and not train_loader.dataset.switched and epoch >= switch_epoch:
            train_loader.dataset.switch_pipeline()
            if hasattr(train_loader, 'persistent_workers') and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                restart_dataloader = True
            LOGGER.info(f'======= Training data pipeline is switched to stage2 at epoch {epoch}! ==========')
        else:
            if restart_dataloader:
                train_loader._DataLoader__initialized = True

        # -- pbar and meter ✅
        pbar = enumerate(train_loader)
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)  # progress bar
            METERS.reset()

        # -- optimizer ✅
        optimizer_wrapper.zero_grad()

        model.train()
        # 🔶🔶 Start train epoch 🔶🔶
        for idx, data_batch in pbar:  # batch -------------------------------------------------------------
            # 🔷🔷 Before train iter 🔷🔷
            # -- get learning rate ✅
            # Dict {'lr': List, (Optional)'base_lr': List}
            lr_dict = optimizer_wrapper.get_lr()
            assert isinstance(lr_dict, dict), 'optim_wrapper.get_lr()` should return a dict of learning rate'
            # # -- get timer ☑️
            # date_time = time.time() - epoch_time

            # 🔷🔷 Start train iter 🔷🔷
            # -- Preprocess ✅
            # data_batch: Dict, {'inputs': imgs, 'data_samples': label}
            # imgs = [batch, 3, h, w] in BGR int8 format
            # data_preprocessor() doing following steps
            # (1) image normalization, including
            #       - convert BGR to RGB(if bgr_to_rgb=True in data_preprocessor setting)
            #       - img.float()
            #       - (img-mean)/std
            # (2) moving all images and labels into GPU
            if type(model) in [nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]:
                data = model.module.data_preprocessor(data = data_batch, training = True)
            else:
                data = model.data_preprocessor(data = data_batch, training = True)

            # -- Forward ✅
            # Normal model forward, nothing special
            # losses is a Dict, may have multiple losses from different loss functions.
            losses = model(**data, mode = 'loss')
            # Merge all losses
            # parsed_loss: final loss goes to optimizer, just one value
            # loss_log_vars: for log purpose only, Dict, {'different loss': Tensor, ..., 'acc_pose': Tensor}
            if type(model) in [nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]:
                parsed_loss, loss_log_vars = model.module.parse_losses(losses)
            else:
                parsed_loss, loss_log_vars = model.parse_losses(losses)
            
            # if RANK != -1:
            #     # Explaination: https://github.com/ultralytics/ultralytics/issues/6555   
            #     parsed_loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # -- Backward ✅
            # update_params() do following steps
            # (1) loss.backward()
            # (2) check accumulate step
            # (3) if accumulate,
            #        - do gradient clip
            #        - optimizer.step()
            #        - optimizer.zero_grad()
            #     else:
            #        - just record some accumulation and training information
            optimizer_wrapper.update_params(parsed_loss)

            # 🔷🔷 After train iter 🔷🔷 ☑️
            # # -- timer ☑️
            # # hooks/runtime_info_hook.py
            # iter_time = time.time() - epoch_time
            # epoch_time = time.time()
            # time_sec_tot += iter_time
            # time_sec_avg = time_sec_tot / (idx - start_iter + 1)
            # eta = time_sec_avg * (max_iters - idx - 1)
            
            # -- Log  ✅
            if RANK in {-1, 0}:
                lr_str = ''
                for lr_name, lr_value in lr_dict.items():
                    lr_str += f'{lr_name}: ' + ','.join(f'{lr:.4e}' for lr in lr_value)
                mem_str = f'mem: {torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                loss_str = ''
                for loss_name, loss_value in loss_log_vars.items():
                    loss_str += f'{loss_name}: {loss_value.item():.5f} ' 
                log_str = f'Epoch:{epoch}/{max_epochs - 1} - {mem_str}|{lr_str}|{loss_str}' # progress bar showing real time loss, not mean loss
                pbar.set_description(log_str)
                
                METERS.update(loss_log_vars) # meter records mean losses, mean accuracy

            # -- Scheduler update(Iter Level) ✅
            for scheduler in schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()
                
            # end batch/iter ------------------------------------------------------------------------------------------------

        # 🔶🔶 After train epoch 🔶🔶 ☑️
        # -- Logger for current epoch ✅
        if RANK in {-1, 0}:
            loss_log_vars = METERS.get_avg_info()
            loss_str = ''
            for loss_name, loss_value in loss_log_vars.items():
                loss_str += f'(epoch_avg){loss_name}: {loss_value.item()} '
            log_str = f'Epoch-[{epoch}/{max_epochs - 1}] - {mem_str}|(epoch_end){lr_str}|{loss_str}' # logger record mean loss and acc from training
            LOGGER.info(f'(Train){log_str}')
        
        # -- Scheduler update(Epoch Level) ✅
        for scheduler in schedulers:
            if scheduler.by_epoch:
                scheduler.step()

        # -- Evaluation and save model on epoch end ☑️
        if RANK in {-1, 0}:
            # Evaluation ✅
            results = evaluate(model = model,
                               dataloader = val_loader,
                               metric = evaluate_metric,
                               )
            results_str = ', '.join([f'{k}: {v:.5f}' for k,v in results.items()])
            LOGGER.info(f'(Val){results_str}')
            # Update best mAP ✅
            # results = 'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)'
            mAP = results['coco/AP']
            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch
            LOGGER.info(f'(Val)Current best mAP: {best_mAP} on epoch {best_epoch}')

            # Save model ✅
            if args.quant:
                quantizer.save_model(model_path=os.path.join(args.save_dir, 'last.pth'))
                if best_mAP == mAP:
                    quantizer.save_model(model_path=os.path.join(args.save_dir, 'best.pth'))
            else:
                if type(model) in [nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                ckpt = {'epoch': epoch,
                        'best_mAP': [best_mAP, best_epoch],
                        'model': model_state_dict, 
                        'optimizer_wrapper': optimizer_wrapper.state_dict(),
                        'schedulers': [sch.state_dict() for sch in schedulers],
                        'date': datetime.now().isoformat()
                        }
                # Save last, best and delete
                torch.save(ckpt, os.path.join(args.save_dir, 'last.pth'))
                if best_mAP == mAP:
                    torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
                del ckpt
                    
        # # -- EarlyStopping ☑️
        # if RANK != -1:  # if DDP training
        #     broadcast_list = [stop if RANK == 0 else None]
        #     dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
        #     if RANK != 0:
        #         stop = broadcast_list[0]
        # if stop:
        #     break  # must break all DDP ranks

        torch.cuda.empty_cache()
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    # 💎💎 After train 💎💎
    # -- log ✅
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - training_start_time) / 3600:.3f} hours.')

    # -- Final Evaluation using best weight  ✅
    if RANK in {-1, 0} and not args.quant:
        best_weight_path = os.path.join(args.save_dir, 'best.pth')
        ww = torch.load(best_weight_path, map_location='cpu')['model']
        LOGGER.info(f'\nValidating on best checkpoint {best_weight_path}...')

        model = build_pose_estimator(configs['model'])
        model.load_state_dict(ww)
        model = model.to(device)
        
        results = evaluate(model = model,
                           dataloader = val_loader,
                           metric = evaluate_metric,
                           )
        
        results_str = ', '.join([f'{k}: {v:.4f}' for k,v in results.items()])
        LOGGER.info(f'(Final Val){results_str}')
        
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='model config py file')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='resume training from ckeckpoint')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--save_dir', default='', help='save to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--quant', action='store_true', help='enable QAT')
    
    return parser.parse_args()



def main(args):
    # assert (args.cfg_file is None and args.pretrained_weight is None)^(args.resume_ckpt is None),\
    #     'pass only one of checkpoint or cfg_file with pretrained_weight'
    
    if args.save_dir == '':
        args.save_dir = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = os.path.join('work_dirs', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.resume_ckpt:
        assert os.path.isdir(args.resume_ckpt), f'checkpoint {args.resume_ckpt}, should be a directory'
        files = os.listdir(args.resume_ckpt)
        for f in files:
            if f.endswith('.py'):
                args.cfg_file = os.path.join(args.resume_ckpt, f)
                break
        args.checkpoint = os.path.join(args.resume_ckpt, 'last.pth')
        open(os.path.join(args.save_dir,f'resumed_from_checkpoint_{args.resume_ckpt.rstrip("/").split("/")[-1]}'), 'w').close()

    configs = parse_config_file(args.cfg_file)
    
    if RANK in {-1, 0}:
        args_string = print_args(vars(args))
        LOGGER.info(args_string)

    # DDP mode ✅
    batch_size = configs['train_dataloader']['batch_size']
    device, msg = select_device(args.device, batch_size=batch_size)
    if RANK in {-1, 0}:
        LOGGER.info(msg)
        
    if LOCAL_RANK != -1:
        assert batch_size % WORLD_SIZE == 0, f'Training batch size must be multiple of WORLD_SIZE, but got {batch_size}'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        torch.distributed.init_process_group(backend = 'nccl' if torch.distributed.is_nccl_available() else 'gloo',
                                             timeout = timedelta(seconds=10800))

    # Train ☑️
    train(args, configs, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)


"""
** Single GPU training
python train.py --cfg_file configs/simcc/coco/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192.py
python train.py --cfg_file configs/my_custom/udp_mobilenetv2_b128-210e_aic-coco-192x192.py
python train.py --cfg_file configs/my_custom/reg_mobilenetv2_rle_b256_420e_aic-coco-192x192.py

python train.py --cfg_file configs/my_custom/reg_mobilenetv2_rle_b256_aic-coco-192x192_quant.py --quant

** Resume training
python train.py --resume_ckpt work_dirs/202402041807/

** Multi GPU DDP training
python -m torch.distributed.run --nproc_per_node 4(需要使用的GPU数量) --master_port 2345(init_process_group卡死就换port) train.py --cfg_file configs/simcc/coco/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192.py(正常传参数) --device 0,1,2,3 
"""
