import pdb
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset_wrappers import CombinedDataset, ConcatDataset, RepeatDataset
from .datasets import *
from .samplers import DefaultSampler, InfiniteSampler, MultiSourceSampler
from .collate_fn import pseudo_collate, default_collate

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def _concat_dataset(cfg, default_args=None):
    types = cfg['type']
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    dataset_infos = cfg.get('dataset_info', None)

    num_joints = cfg['data_cfg'].get('num_joints', None)
    dataset_channel = cfg['data_cfg'].get('dataset_channel', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['ann_file'] = ann_files[i]

        if isinstance(types, (list, tuple)):
            cfg_copy['type'] = types[i]
        if isinstance(img_prefixes, (list, tuple)):
            cfg_copy['img_prefix'] = img_prefixes[i]
        if isinstance(dataset_infos, (list, tuple)):
            cfg_copy['dataset_info'] = dataset_infos[i]

        if isinstance(num_joints, (list, tuple)):
            cfg_copy['data_cfg']['num_joints'] = num_joints[i]

        if isinstance(dataset_channel, (list,tuple)) and isinstance(dataset_channel[0], list):
            cfg_copy['data_cfg']['dataset_channel'] = dataset_channel[i]

        dataset_class = eval(cfg_copy.pop('type'))
        default_args = {} if default_args is None else default_args
        default_args.update(cfg_copy)
        tmp = dataset_class(**default_args)
        datasets.append(tmp)

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset(cfg)
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(cfg['datasets'])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(cfg['dataset'], cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        cfg_copy = copy.deepcopy(cfg)
        dataset_class = eval(cfg_copy.pop('type'))
        default_args = {} if default_args is None else default_args
        default_args.update(cfg_copy)
        dataset = dataset_class(**default_args)
    return dataset

def build_sampler(cfg, dataset, seed=None):
    """
    cfg: Dict
    dataset: torch.utils.data.Dataset
    seed: int
    """
    sampler_cfg = copy.deepcopy(cfg)
    sampler_class = eval(sampler_cfg.pop('type'))
    sampler = sampler_class(dataset = dataset,
                            seed = seed,
                            **sampler_cfg)
    return sampler

def build_collate_fn(cfg):
    collate_cfg = copy.deepcopy(cfg)
    collate_fn = eval(collate_cfg.pop('type'))
    collate_fn = partial(collate_fn, **collate_cfg)
    return collate_fn

def build_dataloader(cfg):
    dataloader_cfg = copy.deepcopy(cfg)
    
    dataset_cfg = dataloader_cfg.pop('dataset')
    dataset = build_dataset(dataset_cfg)

    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler = build_sampler(sampler_cfg, dataset)
    
    collate_fn_cfg = dataloader_cfg.pop('collate_fn', dict(type='pseudo_collate'))
    collate_fn = build_collate_fn(collate_fn_cfg)
    
    dataloader = DataLoader(dataset = dataset,
                            sampler = sampler,
                            batch_sampler = None,
                            collate_fn = collate_fn,
                            worker_init_fn = None,
                            **dataloader_cfg)
    return dataloader, dataset


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
