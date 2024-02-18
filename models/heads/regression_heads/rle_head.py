from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..base_head import BaseHead
from ...utils.tta import flip_coordinates
from ...loss import build_loss
from codec import build_codec
from evaluation.functional import keypoint_pck_accuracy


class RLEHead(BaseHead):
    """Top-down regression head introduced in `RLE`_ by Li et al(2021). The
    head is composed of fully-connected layers to predict the coordinates and
    sigma(the variance of the coordinates) together.

    Args:
        in_channels (int | sequence[int]): Number of input channels
        num_joints (int): Number of joints
        loss (Config): Config for keypoint loss.
                       Defaults to use: class:`RLELoss`.
        decoder (Config, optional): The decoder config that controls decoding
                                    keypoint coordinates from the network output.
                                    Defaults to `None`.
        init_cfg (Config, optional): Config to control the initialization. 
                                     See: attr:`default_init_cfg` for default settings.

    .. _`RLE`: https://arxiv.org/abs/2107.11291
    """
    _version = 2.5

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 num_joints: int,
                 loss = dict(type='RLELoss', use_target_weight=True),
                 decoder: dict = None,
                 init_cfg: dict = None):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.loss_module = build_loss(loss)
        if decoder is not None:
            self.decoder = build_codec(decoder)
        else:
            self.decoder = None

        # Define fully-connected layers
        # ***VERY IMPORTANT***
        # Data distribution of coords and sigma are different,
        # For quantization purpose, DO NOT merge coords and sigma into 1 FC layer,
        # otherwise the quantization of the FC layer is gonna be terrible.
        self.fc_coord = nn.Linear(in_channels, self.num_joints * 2)
        self.fc_sigma = nn.Linear(in_channels, self.num_joints * 2)

    def forward(self, feats: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.
        Args:
            feats (Tensor, Tuple[Tensor]): Feature map/Multi scale feature maps.
                                           Some backbones will output multiple feature maps, but
                                           RLE only need the last one.
        Returns:
            Tensor: output coordinates and sigmas.
        """
        if isinstance(feats, tuple):
            x = feats[-1]
        else:
            x = feats

        x = torch.flatten(x, 1)

        coords = self.fc_coord(x)  # [N, 34]
        sigmas = self.fc_sigma(x)  # [N, 34]

        return coords.reshape(-1, self.num_joints, 2), sigmas.reshape(-1, self.num_joints, 2)

    def predict(self, feats: Union[Tensor, Tuple[Tensor]], batch_data_samples, test_cfg = {}):
        """Predict results from outputs."""

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']

            _feats, _feats_flip = feats

            _batch_coords, _batch_sigmas = self.forward(_feats)
            _batch_sigmas = _batch_sigmas.sigmoid()
            _batch_coords = torch.cat((_batch_coords, _batch_sigmas), dim=-1) # [B, K, 4]

            _batch_coords_flip, _batch_sigmas_flip = self.forward(_feats_flip)
            _batch_coords_flip = torch.cat((_batch_coords_flip, _batch_sigmas_flip), dim=-1)
            _batch_coords_flip = flip_coordinates(_batch_coords_flip,
                                                  flip_indices=flip_indices,
                                                  shift_coords=test_cfg.get('shift_coords', True),
                                                  input_size=input_size)
            _batch_coords_flip[..., 2:] = _batch_coords_flip[..., 2:].sigmoid()

            # merge original and flipped, use mean value as final result
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
            batch_sigmas = (_batch_sigmas + _batch_sigmas_flip) * 0.5
        else:
            batch_coords, batch_sigmas = self.forward(feats)  # (B, K, 2)
            batch_sigmas = batch_sigmas.sigmoid()
            batch_coords = torch.cat((batch_coords, batch_sigmas),dim=-1) # [B, K, 4]

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        return preds

    def loss(self, inputs: Union[Tensor, Tuple[Tensor]], batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords, pred_sigmas = self.forward(inputs)

        keypoint_labels = torch.cat([d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_coords, pred_sigmas,
                                keypoint_labels, keypoint_weights.unsqueeze(-1))

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(pred=pred_coords.detach().cpu().numpy(),
                                              gt=keypoint_labels.detach().cpu().numpy(),
                                              mask=(keypoint_weights.detach().cpu().numpy()) > 0,
                                              thr=0.05,
                                              norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Linear'], std=0.01, bias=0)]
        return init_cfg
