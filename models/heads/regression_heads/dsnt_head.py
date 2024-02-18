# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .integral_regression_head import IntegralRegressionHead
from evaluation.functional import keypoint_pck_accuracy

OptIntSeq = Optional[Sequence[int]]


class DSNTHead(IntegralRegressionHead):
    """Top-down integral regression head introduced in `DSNT`_ by Nibali et
    al(2018). The head contains a differentiable spatial to numerical transform
    (DSNT) layer that do soft-argmax operation on the predicted heatmaps to
    regress the coordinates.

    This head is used for algorithms that require supervision of heatmaps
    in `DSNT` approach.

    Args:
        in_channels (int | sequence[int]): Number of input channels
        in_featuremap_size (int | sequence[int]): Size of input feature map
        num_joints (int): Number of joints
        lambda_t (int): Discard heatmap-based loss when current
            epoch > lambda_t. Defaults to -1.
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
            :class:`DSNTLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`DSNT`: https://arxiv.org/abs/1801.07372
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
                 lambda_t: int = -1,
                 debias: bool = False,
                 beta: float = 1.0,
                 deconv_out_channels = (256, 256, 256),
                 deconv_kernel_sizes = (4, 4, 4),
                 conv_out_channels = None,
                 conv_kernel_sizes = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: dict = dict(type='MultipleLossWrapper',
                                   losses=[dict(type='SmoothL1Loss', use_target_weight=True),
                                           dict(type='JSDiscretLoss', use_target_weight=True)]
                                   ),
                 decoder: dict = None,
                 init_cfg: dict = None):

        super().__init__(in_channels=in_channels,
                         in_featuremap_size=in_featuremap_size,
                         num_joints=num_joints,
                         debias=debias,
                         beta=beta,
                         deconv_out_channels=deconv_out_channels,
                         deconv_kernel_sizes=deconv_kernel_sizes,
                         conv_out_channels=conv_out_channels,
                         conv_kernel_sizes=conv_kernel_sizes,
                         final_layer=final_layer,
                         loss=loss,
                         decoder=decoder,
                         init_cfg=init_cfg)

        self.lambda_t = lambda_t

    def loss(self, inputs, batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_coords, pred_heatmaps = self.forward(inputs)
        keypoint_labels = torch.cat([d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])

        input_list = [pred_coords, pred_heatmaps]
        target_list = [keypoint_labels, gt_heatmaps]
        # calculate losses
        losses = dict()

        loss_list = self.loss_module(input_list, target_list, keypoint_weights)

        loss = loss_list[0] + loss_list[1]

        if self.lambda_t > 0:
            mh = MessageHub.get_current_instance()
            cur_epoch = mh.get_info('epoch')
            if cur_epoch >= self.lambda_t:
                loss = loss_list[0]

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
