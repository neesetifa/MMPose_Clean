# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, List, Optional, Sized, Union

import torch
from torch.utils.data import Sampler

from utils.dist import get_rank, RANK, WORLD_SIZE, sync_random_seed
from .dataset_wrappers import CombinedDataset

RANK = 0 if RANK==-1 else RANK 

class DefaultSampler(Sampler):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        self.rank = RANK # get_rank()
        self.world_size = WORLD_SIZE

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.rank) / self.world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class InfiniteSampler(Sampler):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        self.rank = RANK
        self.world_size = WORLD_SIZE
        
        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        yield from self.indices

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass
        

class MultiSourceSampler(Sampler):
    """Multi-Source Sampler. According to the sampling ratio, sample data from
    different datasets to form batches.

    Args:
        dataset (Sized): The dataset
        batch_size (int): Size of mini-batch
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch
        shuffle (bool): Whether shuffle the dataset or not. Defaults to
            ``True``
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
        seed (int, optional): Random seed. If ``None``, set a random seed.
            Defaults to ``None``
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 round_up: bool = True,
                 seed: Optional[int] = None) -> None:

        assert isinstance(dataset, CombinedDataset),\
            f'The dataset must be CombinedDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            f'batch_size must be a positive integer value, but got batch_size={batch_size}'
        assert isinstance(source_ratio, list), \
            f'source_ratio must be a list, but got source_ratio={source_ratio}'
        assert len(source_ratio) == len(dataset._lens), \
            f'The length of source_ratio must be equal to the number of datasets, but got source_ratio={source_ratio}'

        self.rank = RANK
        self.world_size = WORLD_SIZE

        self.dataset = dataset
        self.cumulative_sizes = [0] + list(itertools.accumulate(dataset._lens))
        self.batch_size = batch_size
        self.source_ratio = source_ratio
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.num_per_source = [int(batch_size * sr / sum(source_ratio)) for sr in source_ratio]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == batch_size, \
            f'The sum of num_per_source must be equal to batch_size, but get {self.num_per_source}'

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.round_up = round_up
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        num_iters = self.num_samples // self.batch_size
        if self.round_up and self.num_samples > num_iters * self.batch_size:
            num_iters += 1
        for i in range(num_iters):
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
        return iter(batch_buffer)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Compatible in `epoch-based runner."""
        pass
