# Copyright (c) OpenMMLab. All rights reserved.

from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union, Sequence

import numpy as np
from torch.utils.data import Dataset
from .datasets.utils import parse_pose_metainfo
from .datasets import AicDataset, CocoDataset
from .transforms import Compose


class CombinedDataset(Dataset):
    """A wrapper of combined dataset.

    Args:
        metainfo (dict): The meta information of combined dataset.
        datasets (list): The configs of datasets to be combined.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        sample_ratio_factor (list, optional): A list of sampling ratio
            factors for each dataset. Defaults to None
    """

    def __init__(self,
                 metainfo: dict,
                 datasets: list,
                 pipeline: List[Union[dict, Callable]] = [],
                 pipeline_stage2: List[Union[dict, Callable]] = [],
                 sample_ratio_factor: Optional[List[float]] = None,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        self.datasets = []
        self.resample = sample_ratio_factor is not None

        for cfg in datasets:
            cfg_copy = deepcopy(cfg)
            dataset_class = eval(cfg_copy.pop('type'))
            dataset = dataset_class(**cfg_copy)
            self.datasets.append(dataset)

        self._lens = [len(dataset) for dataset in self.datasets]
        if self.resample:
            assert len(sample_ratio_factor) == len(datasets), f'the length ' \
                f'of `sample_ratio_factor` {len(sample_ratio_factor)} does ' \
                f'not match the length of `datasets` {len(datasets)}'
            assert min(sample_ratio_factor) >= 0.0, 'the ratio values in ' \
                '`sample_ratio_factor` should not be negative.'
            self._lens_ori = self._lens
            self._lens = [
                round(l * sample_ratio_factor[i])
                for i, l in enumerate(self._lens_ori)
            ]

        self._len = sum(self._lens)
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        
        self.pipeline_stage1 = Compose(pipeline)
        self.pipeline = self.pipeline_stage1
        if pipeline_stage2:
            self.pipeline_stage2 = Compose(pipeline_stage2)
        else:
            self.pipeline_stage2 = None
        self.switched = False
        
        self._metainfo = parse_pose_metainfo(metainfo)
        # Full initialize the dataset.
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def switch_pipeline(self):
        if self.pipeline_stage2 is not None:
            self.pipeline = self.pipeline_stage2
            self.switched = True
        else:
            raise ValueError('No stage2 pipeline provided')
            
    @property
    def metainfo(self):
        return deepcopy(self._metainfo)

    def __len__(self):
        return self._len

    def _get_subset_index(self, index: int) -> Tuple[int, int]:
        """Given a data sample's global index, return the index of the sub-
        dataset the data sample belongs to, and the local index within that
        sub-dataset.

        Args:
            index (int): The global data sample index

        Returns:
            tuple[int, int]:
            - subset_index (int): The index of the sub-dataset
            - local_index (int): The index of the data sample within
                the sub-dataset
        """
        if index >= len(self) or index < -len(self):
            raise ValueError(
                f'index({index}) is out of bounds for dataset with '
                f'length({len(self)}).')

        if index < 0:
            index = index + len(self)

        subset_index = 0
        while index >= self._lens[subset_index]:
            index -= self._lens[subset_index]
            subset_index += 1

        if self.resample:
            gap = (self._lens_ori[subset_index] -
                   1e-4) / self._lens[subset_index]
            index = round(gap * index + np.random.rand() * gap - 0.5)

        return subset_index, index

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.The source dataset is
        depending on the index.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """

        data_info = self.get_data_info(idx)

        # the assignment of 'dataset' should not be performed within the
        # `get_data_info` function. Otherwise, it can lead to the mixed
        # data augmentation process getting stuck.
        data_info['dataset'] = self

        return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``CombinedDataset``.
        Returns:
            dict: The idx-th annotation of the datasets.
        """
        subset_idx, sample_idx = self._get_subset_index(idx)
        # Get data sample processed by ``subset.pipeline``
        data_info = self.datasets[subset_idx][sample_idx]

        if 'dataset' in data_info:
            data_info.pop('dataset')

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices'
        ]

        for key in metainfo_keys:
            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def full_init(self):
        """Fully initialize all sub datasets."""

        if self._fully_initialized:
            return

        for dataset in self.datasets:
            dataset.full_init()
        self._fully_initialized = True

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')



class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Note:
        ``RepeatDataset`` should not inherit from ``Dataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``RepeatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``Dataset``.

    Args:
        dataset (Dataset or dict): The dataset to be repeated.
        times (int): Repeat times.
        lazy_init (bool): Whether to load annotation during
            instantiation. Defaults to False.
    """

    def __init__(self,
                 dataset: Union[Dataset, dict],
                 times: int,
                 lazy_init: bool = False):
        self.dataset: Dataset
        if isinstance(dataset, dict):
            dataset_class = eval(dataset.pop('type'))
            self.dataset = dataset_class(**dataset)
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`Dataset` instance, but got {type(dataset)}')
        self.times = times
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    #@force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx: Global index of ``RepeatDataset``.

        Returns:
            idx (int): Local index of data.
        """
        return idx % self._ori_len

    #@force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate the '
                'speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset[sample_idx]

    #@force_full_init
    def __len__(self):
        return self.times * self._ori_len

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`RepeatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `RepeatDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> 'Dataset':
        """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`RepeatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `RepeatDataset`.')


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset`` and support lazy_init.

    Note:
        ``ConcatDataset`` should not inherit from ``Dataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``Dataset``.

    Args:
        datasets (Sequence[Dataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        ignore_keys (List[str] or str): Ignore the keys that can be
            unequal in `dataset.metainfo`. Defaults to None.
            `New in version 0.3.0.`
    """

    def __init__(self,
                 datasets: Sequence[Union[Dataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None):
        self.datasets: List[Dataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                dataset_copy = deepcopy(dataset)
                dataset_class = eval(dataset_copy.pop('type'))
                tmp = dataset_class(**dataset_copy)
                self.datasets.append(tmp)
            elif isinstance(dataset, Dataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`Dataset` instance, but got {type(dataset)}')
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    raise ValueError(
                        f'{key} does not in the meta information of '
                        f'the {i}-th dataset')
                first_type = type(self._metainfo[key])
                cur_type = type(dataset.metainfo[key])
                if first_type is not cur_type:  # type: ignore
                    raise TypeError(
                        f'The type {cur_type} of {key} in the {i}-th dataset '
                        'should be the same with the first dataset '
                        f'{first_type}')
                if (isinstance(self._metainfo[key], np.ndarray)
                        and not np.array_equal(self._metainfo[key],
                                               dataset.metainfo[key])
                        or self._metainfo[key] != dataset.metainfo[key]):
                    raise ValueError(
                        f'The meta information of the {i}-th dataset does not '
                        'match meta information of the first dataset')

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the first dataset in ``self.datasets``.

        Returns:
            dict: Meta information of first dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for d in self.datasets:
            d.full_init()
        # Get the cumulative sizes of `self.datasets`. For example, the length
        # of `self.datasets` is [2, 3, 4], the cumulative sizes is [2, 5, 9]
        super().__init__(self.datasets)
        self._fully_initialized = True

    #@force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f'absolute value of index({idx}) should not exceed dataset'
                    f'length({len(self)}).')
            idx = len(self) + idx
        # Get `dataset_idx` to tell idx belongs to which dataset.
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # Get the inner index of single dataset.
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    #@force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)

    #@force_full_init
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ConcatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`ConcatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ConcatDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> 'Dataset':
        """Not supported in ``ConcatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            '`ConcatDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ConcatDataset`.')
