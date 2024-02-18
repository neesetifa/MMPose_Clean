# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union, Dict

import torch
from torch import Tensor, nn

from ..base_head import BaseHead
from ...utils.tta import flip_heatmaps
from ...loss import build_loss
from codec import build_codec
from evaluation.functional import pose_pck_accuracy
from structures import PixelData

OptIntSeq = Optional[Sequence[int]]


class HeatmapHead(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels = (256, 256, 256),
                 deconv_kernel_sizes = (4, 4, 4),
                 conv_out_channels = None,
                 conv_kernel_sizes = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: Dict= dict(type='KeypointMSELoss', use_target_weight=True),
                 decoder: Dict = None,
                 init_cfg: Dict = None):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = build_loss(loss)
        if decoder is not None:
            self.decoder = build_codec(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels)!=len(deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(in_channels=in_channels,
                                                          layer_out_channels=deconv_out_channels,
                                                          layer_kernel_sizes=deconv_kernel_sizes,
                                                          )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(in_channels=in_channels,
                                                      layer_out_channels=conv_out_channels,
                                                      layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            # cfg = dict(
            #     type='Conv2d',
            #     in_channels=in_channels,
            #     out_channels=out_channels,
            #     kernel_size=1)
            # cfg.update(final_layer)
            # self.final_layer = build_conv_layer(cfg)
            self.final_layer = nn.Conv2d(in_channels = in_channels,
                                         out_channels = out_channels,
                                         kernel_size = final_layer['kernel_size'])
        else:
            self.final_layer = nn.Identity()

        self.init_cfg = self.default_init_cfg if init_cfg is None else init_cfg

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            conv_layer = nn.Conv2d(in_channels = in_channels,
                                   out_channels = out_channels,
                                   kernel_size = kernel_size,
                                   stride = 1,
                                   padding = (kernel_size - 1) // 2)
            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')

            deconv_layer = nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=2,
                                              padding=padding,
                                              output_padding=output_padding,
                                              bias=False)
            layers.append(deconv_layer)
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, feats: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Feature map/Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        if isinstance(feats, tuple):
            x = feats[-1]
        else:
            x = feats

        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)

        return x

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
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(self.forward(_feats_flip),
                                                 flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                                                 flip_indices=flip_indices,
                                                 shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        else:
            return preds

    def loss(self, feats, batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(output=pred_fields.detach().cpu().numpy(),
                                              target=gt_heatmaps.detach().cpu().numpy(),
                                              mask=keypoint_weights.detach().cpu().numpy() > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [dict(type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
                    dict(type='Constant', layer='BatchNorm2d', val=1)
                    ]
        return init_cfg
