# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..base_head import BaseHead
from ..heatmap_heads import HeatmapHead
from ...utils.tta import flip_coordinates, flip_heatmaps
from ...loss import build_loss
from codec import build_codec
from evaluation.functional import keypoint_pck_accuracy
from structures import PixelData


OptIntSeq = Optional[Sequence[int]]


class IntegralRegressionHead(BaseHead):
    """Top-down integral regression head introduced in `IPR`_ by Xiao et
    al(2018). The head contains a differentiable spatial to numerical transform
    (DSNT) layer that do soft-argmax operation on the predicted heatmaps to
    regress the coordinates.

    This head is used for algorithms that only supervise the coordinates.

    Args:
        in_channels (int | sequence[int]): Number of input channels
        in_featuremap_size (int | sequence[int]): Size of input feature map
        num_joints (int): Number of joints
        debias (bool): Whether to remove the bias of Integral Pose Regression.
            see `Removing the Bias of Integral Pose Regression`_ by Gu et al
            (2021). Defaults to ``False``.
        beta (float): A smoothing parameter in softmax. Defaults to ``1.0``.
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config for keypoint loss. Defaults to use
            :class:`SmoothL1Loss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`IPR`: https://arxiv.org/abs/1711.08229
    .. _`Debias`:
    """

    _version = 2.5

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
                 debias: bool = False,
                 beta: float = 1.0,
                 deconv_out_channels= (256, 256, 256),
                 deconv_kernel_sizes = (4, 4, 4),
                 conv_out_channels = None,
                 conv_kernel_sizes = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: dict = dict(type='SmoothL1Loss', use_target_weight=True),
                 decoder: dict = None,
                 init_cfg: dict = None):

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.debias = debias
        self.beta = beta
        self.loss_module = build_loss(loss)
        if decoder is not None:
            self.decoder = build_codec(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple([s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.simplebaseline_head = HeatmapHead(in_channels=in_channels,
                                                   out_channels=num_joints,
                                                   deconv_out_channels=deconv_out_channels,
                                                   deconv_kernel_sizes=deconv_kernel_sizes,
                                                   conv_out_channels=conv_out_channels,
                                                   conv_kernel_sizes=conv_kernel_sizes,
                                                   final_layer=final_layer)

            if final_layer is not None:
                in_channels = num_joints
            else:
                in_channels = deconv_out_channels[-1]

        else:
            self.simplebaseline_head = None

            if final_layer is not None:
                self.final_layer = nn.Conv2d(in_channels = in_channels,
                                             out_channels = out_channels,
                                             kernel_size = final_layer['kernel_size'])
            else:
                self.final_layer = None

            self.heatmap_size = in_featuremap_size

        if isinstance(in_channels, list):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        W, H = self.heatmap_size
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, 1, W) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H, 1) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

    def _linear_expectation(self, heatmaps: Tensor, linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""

        B, N, _, _ = heatmaps.shape
        heatmaps = heatmaps.mul(linspace).reshape(B, N, -1)
        expectation = torch.sum(heatmaps, dim=2, keepdim=True)

        return expectation

    def _flat_softmax(self, featmaps: Tensor) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""

        _, N, H, W = featmaps.shape

        featmaps = featmaps.reshape(-1, N, H * W)
        heatmaps = F.softmax(featmaps, dim=2)

        return heatmaps.reshape(-1, N, H, W)

    def forward(self, feats: Tuple[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates(and sigmas[optional]).
        """
        if self.simplebaseline_head is None:
            feats = feats[-1]
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.simplebaseline_head(feats)

        heatmaps = self._flat_softmax(feats * self.beta)

        pred_x = self._linear_expectation(heatmaps, self.linspace_x)
        pred_y = self._linear_expectation(heatmaps, self.linspace_y)

        if self.debias:
            B, N, H, W = feats.shape
            C = feats.reshape(B, N, H * W).exp().sum(dim=2).reshape(B, N, 1)
            pred_x = C / (C - 1) * (pred_x - 1 / (2 * C))
            pred_y = C / (C - 1) * (pred_y - 1 / (2 * C))

        coords = torch.cat([pred_x, pred_y], dim=-1)
        return coords, heatmaps

    def predict(self, feats, batch_data_samples, test_cfg = {}):
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords, _batch_heatmaps = self.forward(_feats)

            _batch_coords_flip, _batch_heatmaps_flip = self.forward(_feats_flip)
            _batch_coords_flip = flip_coordinates(_batch_coords_flip,
                                                  flip_indices=flip_indices,
                                                  shift_coords=test_cfg.get('shift_coords', True),
                                                  input_size=input_size)
            _batch_heatmaps_flip = flip_heatmaps(_batch_heatmaps_flip,
                                                 flip_mode='heatmap',
                                                 flip_indices=flip_indices,
                                                 shift_heatmap=test_cfg.get('shift_heatmap', False))

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_coords, batch_heatmaps = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds

    def loss(self, inputs, batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords, _ = self.forward(inputs)
        keypoint_labels = torch.cat([d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()

        # TODO: multi-loss calculation
        loss = self.loss_module(pred_coords, keypoint_labels, keypoint_weights)

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
