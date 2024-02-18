# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Tuple, Union, Dict, Optional
from collections import OrderedDict

import torch
from torch import Tensor

from ..backbones import build_backbone
from ..data_preprocessor import BaseDataPreprocessor, build_preprocessor
from ..necks import build_neck
from ..heads import build_head
from ..loss import build_loss

from datasets.datasets.utils import parse_pose_metainfo

class BasePoseEstimator(torch.nn.Module):
    """Base class for pose estimator models.

    Args:
        data_preprocessor (dict | ConfigDict, optional): The pre-processing
            config of :class:`BaseDataPreprocessor`. Defaults to ``None``
        init_cfg (dict | ConfigDict): The model initialization config.
            Defaults to ``None``
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to False.
    """
    _version = 2.5

    def __init__(self,
                 backbone: Dict,
                 neck: Dict = None,
                 head: Dict = None,
                 train_cfg: Dict = {},
                 test_cfg: Dict = {},
                 data_preprocessor: Dict = None,
                 init_cfg: Dict = None,
                 metainfo: Optional[dict] = None):
        super().__init__()

        self.metainfo = self._load_metainfo(metainfo)
        
        self.train_cfg = train_cfg if train_cfg else {}# Usually {}
        self.test_cfg = test_cfg if test_cfg else {}   # Usually dict(flip_test=True), nothing special

        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        self.data_preprocessor = build_preprocessor(data_preprocessor)
            
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)
            self.head.test_cfg = self.test_cfg.copy()

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, 'head') and self.head is not None

    def forward(self,
                inputs: torch.Tensor, data_samples = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    @abstractmethod
    def loss(self, inputs: Tensor, data_samples):
        """Calculate losses from a batch of inputs and data samples."""
        """Return Dict"""

    @abstractmethod
    def predict(self, inputs: Tensor, data_samples):
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        """Return SampleList"""

    def _forward(self, inputs, data_samples = None)-> Union[Tensor, Tuple[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """

        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)

        return x

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif isinstance(loss_value, list):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

    @staticmethod
    def _load_metainfo(metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            return None

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def init_weights(self):
        self.backbone.init_weights()
        # if self.with_neck and hasattr(self.neck, 'init_weights'):
        #     self.neck.init_weights()
        # if self.with_head and hasattr(self.head, 'init_weights'):
        #     self.head.init_weights()
        
    # ======= Following override functions are ONLY for DataPreprocessor =========
    def to(self, *args, **kwargs) -> torch.nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._set_device(torch.device(device))
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> torch.nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device('cuda', index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def cpu(self, *args, **kwargs) -> torch.nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device('cpu'))
        return super().cpu()

    def _set_device(self, device: torch.device) -> None:
        """Recursively set device for `BaseDataPreprocessor` instance.

        Args:
            device (torch.device): the desired device of the parameters and
                buffers in this module.
        """

        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return
            if device is not None:
                module._device = device

        self.apply(apply_fn)
