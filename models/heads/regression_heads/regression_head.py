# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..base_head import BaseHead
from ...utils.tta import flip_coordinates
from ...loss import build_loss
from codec import build_codec
from evaluation.functional import keypoint_pck_accuracy

OptIntSeq = Optional[Sequence[int]]


class RegressionHead(BaseHead):
    """Top-down regression head introduced in `Deeppose`_ by Toshev et al
    (2014). The head is composed of fully-connected layers to predict the
    coordinates directly.

    Args:
        in_channels (int | sequence[int]): Number of input channels
        num_joints (int): Number of joints
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`SmoothL1Loss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Deeppose`: https://arxiv.org/abs/1312.4659
    """

    _version = 2.5

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 num_joints: int,
                 loss: dict = dict(type='SmoothL1Loss', use_target_weight=True),
                 decoder: dict = None,
                 init_cfg: dict = None):

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.loss_module = build_loss(loss)
        if decoder is not None:
            self.decoder = build_codec(decoder)
        else:
            self.decoder = None

        # Define fully-connected layers
        self.fc = nn.Linear(in_channels, self.num_joints * 2)

    def forward(self, feats: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Feature map/Multi scale feature maps.
                                           Some backbones will output multiple feature maps, but
                                           Regression only need the last one.

        Returns:
            Tensor: output coordinates.
        """
        if isinstance(feats, tuple):
            x = feats[-1]
        else:
            x = feats

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x.reshape(-1, self.num_joints, 2)

    def predict(self, feats: Tuple[Tensor], batch_data_samples, test_cfg = {}):
        """Predict results from outputs."""

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords = self.forward(_feats)
            _batch_coords_flip = flip_coordinates(self.forward(_feats_flip),
                                                  flip_indices=flip_indices,
                                                  shift_coords=test_cfg.get('shift_coords', True),
                                                  input_size=input_size)
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        return preds

    def loss(self, inputs: Tuple[Tensor], batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_outputs = self.forward(inputs)

        keypoint_labels = torch.cat([d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_outputs, keypoint_labels,
                                keypoint_weights.unsqueeze(-1))

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(pred=pred_outputs.detach().cpu().numpy(),
                                              gt=keypoint_labels.detach().cpu().numpy(),
                                              mask=(keypoint_weights.detach().cpu().numpy()) > 0,
                                              thr=0.05,
                                              norm_factor=np.ones((pred_outputs.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg
