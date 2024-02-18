# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Tuple, Union, List
import pdb

import torch
from torch import Tensor

from structures import InstanceData
from ..utils.weight_init import *


class BaseHead(torch.nn.Module):
    """Base head. A subclass should override :meth:`predict` and :meth:`loss`.

    Args:
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to None.
    """

    @abstractmethod
    def forward(self, feats: Tuple[Tensor]):
        """Forward the network."""

    @abstractmethod
    def predict(self, feats, batch_data_samples, test_cfg = {}):
        """Predict results from features."""

    @abstractmethod
    def loss(self, feats, batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    def decode(self, batch_outputs):
        """Decode keypoints from outputs.

        Args:
            batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
                a data batch

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
            decoded pose information of the instances of one data sample.
        """

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args, )
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(batch_outputs, self.decoder.batch_decode)

        else:
            if isinstance(batch_outputs, Tensor):
                batch_output_np = batch_outputs.detach().cpu().numpy()
            elif isinstance(batch_outputs, Union[List,Tuple]):
                batch_output_np = [tuple(_x[None, :].detach().cpu().numpy() for _x in _each)
                                   for _each in zip(*batch_outputs)
                                   ]
            else:
                print('Invalid batch_outputs, debug in base_head.py')
                pdb.set_trace()
            
            batch_keypoints = []
            batch_scores = []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                batch_scores.append(scores)

        preds = [InstanceData(keypoints=keypoints, keypoint_scores=scores)
                 for keypoints, scores in zip(batch_keypoints, batch_scores)
                 ]

        return preds

    def init_weights(self):
        for cfg in self.init_cfg:
            init_class = eval(cfg.pop('type'))
            init_ = init_class(**cfg)
            init_(self)
            
