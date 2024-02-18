# yingyi xu
import warnings
from typing import Optional, Sequence, Tuple, Union

import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..base_head import BaseHead
from ..heatmap_heads import HeatmapHead, ViPNASHead
from ...utils.tta import flip_vectors, flip_coordinates
from ...loss import build_loss
from codec.utils import get_simcc_normalized
from codec import build_codec
from evaluation.functional import simcc_pck_accuracy, keypoint_pck_accuracy
from structures import PixelData


OptIntSeq = Optional[Sequence[int]]


class SimCCRLEHead(BaseHead):
    """
    Hybird Head with SimCC and RLE

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        input_size (tuple): Input image size in shape [w, h]
        in_featuremap_size (int | sequence[int]): Size of input feature map
        simcc_split_ratio (float): Split ratio of pixels
        deconv_type (str, optional): The type of deconv head which should
            be one of the following options:

                - ``'heatmap'``: make deconv layers in `HeatmapHead`
                - ``'vipnas'``: make deconv layers in `ViPNASHead`

            Defaults to ``'Heatmap'``
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        deconv_num_groups (Sequence[int], optional): The group number of each
            deconv layer. Defaults to ``(16, 16, 16)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    """

    _version = 1.5

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 input_size: Tuple[int, int],
                 in_featuremap_size: Tuple[int, int],
                 num_joints: int,
                 simcc_split_ratio: float = 2.0,
                 deconv_type: str = 'heatmap',
                 deconv_out_channels = (256, 256, 256),
                 deconv_kernel_sizes = (4, 4, 4),
                 deconv_num_groups = (16, 16, 16),
                 conv_out_channels = None,
                 conv_kernel_sizes = None,
                 final_layer: dict = dict(kernel_size=1),
                 loss: dict = dict(type='CombinedLoss',
                                   losses = dict(simcc_loss = dict(type='KLDiscretLoss',
                                                                   use_target_weight=True),
                                                 rle_loss = dict(type='RLELoss',
                                                                 use_target_weight=True)
                                                 ),
                                   losses_weight = dict(simcc_loss_weight = 1.0,
                                                        rle_loss_weight = 0.01),
                                   ),
                 decoder: dict = None,
                 init_cfg: dict = None,
                 ):
        super().__init__()

        if deconv_type not in {'heatmap', 'vipnas'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `deconv_type` value'
                f'{deconv_type}. Should be one of '
                '{"heatmap", "vipnas"}')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.num_joints = num_joints
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.loss_module = build_loss(loss)
        if decoder is not None:
            self.decoder = build_codec(decoder)
        else:
            self.decoder = None

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple([s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.deconv_head = self._make_deconv_head(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      deconv_type=deconv_type,
                                                      deconv_out_channels=deconv_out_channels,
                                                      deconv_kernel_sizes=deconv_kernel_sizes,
                                                      deconv_num_groups=deconv_num_groups,
                                                      conv_out_channels=conv_out_channels,
                                                      conv_kernel_sizes=conv_kernel_sizes,
                                                      final_layer=final_layer)

            if final_layer is not None:
                in_channels = out_channels
            else:
                in_channels = deconv_out_channels[-1]

        else:
            # 官方mobilenetv2模型走的这条分支
            self.deconv_head = None

            if final_layer is not None:
                self.final_layer = nn.Conv2d(in_channels = in_channels,
                                             out_channels = out_channels,
                                             kernel_size = final_layer['kernel_size'])
            else:
                self.final_layer = None

            self.heatmap_size = in_featuremap_size

        # Define SimCC layers
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)

        # For Soft-argmax 来源 integral_regression_head.py
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, W) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)

        # Define RLE layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_coord = nn.Linear(in_channels, self.num_joints * 2)
        self.fc_sigma = nn.Linear(in_channels, self.num_joints * 2)

    def _make_deconv_head(self,
                          in_channels: Union[int, Sequence[int]],
                          out_channels: int,
                          deconv_type: str = 'heatmap',
                          deconv_out_channels = (256, 256, 256),
                          deconv_kernel_sizes = (4, 4, 4),
                          deconv_num_groups = (16, 16, 16),
                          conv_out_channels = None,
                          conv_kernel_sizes = None,
                          final_layer: dict = dict(kernel_size=1)
                          ) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        if deconv_type == 'heatmap':
            deconv_head = HeatmapHead(in_channels=self.in_channels,
                                      out_channels=out_channels,
                                      deconv_out_channels=deconv_out_channels,
                                      deconv_kernel_sizes=deconv_kernel_sizes,
                                      conv_out_channels=conv_out_channels,
                                      conv_kernel_sizes=conv_kernel_sizes,
                                      final_layer=final_layer)
        else:
            deconv_head = ViPNASHead(in_channels=in_channels,
                                     out_channels=out_channels,
                                     deconv_out_channels=deconv_out_channels,
                                     deconv_num_groups=deconv_num_groups,
                                     conv_out_channels=conv_out_channels,
                                     conv_kernel_sizes=conv_kernel_sizes,
                                     final_layer=final_layer)

        return deconv_head

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Feature map/Multi scale feature maps.
                                   Some backbones will output multiple feature maps, but
                                   Simcc and RLE only need the last one.

        Returns:
            pred_x (Tensor): 1d representation of x, [batch, 17, input_W*k], k is simcc_split_ratio
            pred_y (Tensor): 1d representation of y, [batch, 17, input_H*k], k is simcc_split_ratio
            pred_rle(Tensor): output of RLE,         [batch, 17, 4=(coord_x,coord_y,sigma_x, sigma_y)]
        """
        # 1. RLE head
        if isinstance(feats, tuple):
            feats = feats[-1]
        # feats [batch, channel, input_H//32, input_W//32] = [batch,1280,6,6]
        avg_feats = self.gap(feats)                       # [batch, 1280, 1, 1]
        avg_feats = avg_feats.view(avg_feats.size(0), -1) # [batch, 1280]
        pred_coords = self.fc_coord(avg_feats)            # [batch, 34]
        pred_sigmas = self.fc_sigma(avg_feats)
        pred_coords = pred_coords.reshape(-1, self.num_joints, 2) # [batch, 17, 2]
        pred_sigmas = pred_sigmas.reshape(-1, self.num_joints, 2)

        # 2. SIMCC head
        if self.deconv_head is None:  
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.deconv_head(feats)
        # feats = [batch, 17, input_H//32, input_W//32]

        # flatten the output heatmap
        x = torch.flatten(feats, 2) # [batch, 17, input_H//32*input_W//32]
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y, pred_coords, pred_sigmas

    def predict(self, feats: Tuple[Tensor], batch_data_samples, test_cfg = {}):
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y, _batch_pred_coords, _batch_pred_sigmas = self.forward(_feats)
            _batch_pred_sigmas = _batch_pred_sigmas.sigmoid()

            _batch_pred_x_flip, _batch_pred_y_flip, _batch_pred_coords_flip, _batch_pred_sigmas_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(_batch_pred_x_flip,
                                                                  _batch_pred_y_flip,
                                                                  flip_indices=flip_indices)

            _batch_pred_coords_flip = flip_coordinates(_batch_pred_coords_flip,
                                                       flip_indices=flip_indices,
                                                       shift_coords=test_cfg.get('shift_coords', True),
                                                       input_size=input_size)
            _batch_pred_sigmas_flip = _batch_pred_sigmas_flip.sigmoid()

            # 合并
            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5

            batch_pred_coords = (_batch_pred_coords + _batch_pred_coords_flip) * 0.5
            batch_pred_sigmas = (_batch_pred_sigmas + _batch_pred_sigmas_flip) * 0.5
            
        else:
            batch_pred_x, batch_pred_y, batch_pred_coords, _batch_pred_sigmas = self.forward(feats)
            batch_pred_sigmas = batch_pred_sigmas.sigmoid()
        # batch_pred_xy 的输出是实数域
        # batch_pred_xy shape = [batch, 17, input_size*simcc_ratio]

        # 使用父类的方法decode(), 先用self.decoder解码, 然后使用InstanceData打包
        # decoder具体实现在simcc_label.py里 decode()
        # 注意 self.decode 和 self.decoder 不是一个东西
        preds = self.decode((batch_pred_x, batch_pred_y, batch_pred_coords, batch_pred_sigmas))  # preds[idx] = [batch,] 仍然是实数域
        
        return preds

    def loss(self, feats, batch_data_samples, train_cfg = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        # pred_xy 的输出是实数域, 值域范围实测约 [-60,40], training的话在loss里做softmax
        # pred_xy shape = [batch, 17, input_size*simcc_ratio]
        pred_x, pred_y, pred_coords, pred_sigmas = self.forward(feats) 
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        
        # calculate losses
        losses = dict()
        
        # 1. Simcc Loss
        # gt_xy 值域范围为 [0, 0.06649], 可以参考 simcc_label.py 里 _generate_gaussian() 函数的注释
        gt_x = torch.cat([d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples],dim=0)
        gt_y = torch.cat([d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples],dim=0)
        
        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)
        loss_simcc = self.loss_module.simcc_loss(pred_simcc, gt_simcc, keypoint_weights)

        # 2. RLE Loss
        gt_reg = torch.cat([d.gt_instance_labels.keypoint_labels for d in batch_data_samples])

        # pred_coords有两种获得方法, 使用Simcc输出的heatmap或者RLE head直接输出
        # (1) do Soft-Argmax
        # F.softmax(pred_x, dim=-1) 值域范围实测约为 [0, 0.11] 超出了gt_xy的值域范围
        score_x = F.softmax(pred_x, dim=-1)#*((2*torch.pi)**0.5*self.decoder.sigma[0]) # [batch, 17, W]
        score_y = F.softmax(pred_y, dim=-1)#*((2*torch.pi)**0.5*self.decoder.sigma[1]) # [batch, 17, H]
        pred_coords_x = score_x.mul(self.linspace_x) # [batch, 17, W]
        pred_coords_y = score_y.mul(self.linspace_y) # [batch, 17, H]
        pred_coords_x = pred_coords_x.sum(dim=-1)    # [batch, 17]
        pred_coords_y = pred_coords_y.sum(dim=-1)    # [batch, 17]
        pred_coords = torch.stack([pred_coords_x, pred_coords_y], dim=-1) # [batch, 17, 2]
        # (2) Directly get from FC
        # pred_coords = pred_coords

        # pred_sigma 由RLE head负责, 不能使用simcc的输出, 分析如下
        """
        值域分析:
        pred_xy 值域范围实测约 [-60,40]
        (1) 直接使用 pred_xy
        torch.max(pred_xy, dim=-1) 值域范围实测约为 [-0.02, 40], sigmoid之后范围[0.5,1], 不符合
        (2) 使用softmax之后的
        score_xy = F.softmax(pred_xy, dim=-1) 值域范围实测约为 [0, 0.11]
        torch.max(score_xy, dim=-1) 值域范围实测约为 [0.0026, 0.11], 不符合
        (3) 使用softmax之后再rescale
        score_xy = F.softmax(pred_xy, dim=-1)*((2*torch.pi)**0.5*sigma 值域范围实测约为 [0, 1.65], 不符合
        
        流模型sigma需求值域范围: 实数域[-88,+∞) 或者(1e-45,1), 不然torch.log会报错
        传入的sigma需要 (0,1-1e-45)

        结论:
        无法从pred_xy 得到符合条件的sigma, sigma需要单独预测
        """
        loss_rle = self.loss_module.rle_loss(pred_coords, pred_sigmas, gt_reg,
                                             keypoint_weights.unsqueeze(-1))

        # 最终loss相加在 mmengine/model/base_model/base_model.py的parse_losses()里完成
        losses.update({'loss/simcc': loss_simcc * self.loss_module.simcc_loss_weight,
                       'loss/rle': loss_rle * self.loss_module.rle_loss_weight,
                       })

        # calculate accuracy
        pred_simcc_np = (pred_simcc[0].detach().cpu().numpy(), pred_simcc[1].detach().cpu().numpy())
        gt_simcc_np = (gt_simcc[0].detach().cpu().numpy(), gt_simcc[1].detach().cpu().numpy())
        _, avg_acc_simcc, _ = simcc_pck_accuracy(output=pred_simcc_np,
                                                 target=gt_simcc_np,
                                                 simcc_split_ratio=self.simcc_split_ratio,
                                                 mask=to_numpy(keypoint_weights) > 0,
                                                 )

        _, avg_acc_rle, _ = keypoint_pck_accuracy(pred=pred_coords.detach().cpu().numpy(),
                                                  gt=gt_reg.detach().cpu().numpy(),
                                                  mask=(keypoint_weights.detach().cpu().numpy()) > 0,
                                                  thr=0.05,
                                                  norm_factor=np.ones((pred_coords.size(0), 2), dtype=np.float32))
        
        acc_pose_simcc = torch.tensor(avg_acc_simcc, device=gt_x.device)
        acc_pose_rle = torch.tensor(avg_acc_rle, device=gt_x.device)
        losses.update(acc_pose_simcc=acc_pose_simcc,
                      acc_pose_rle=acc_pose_rle)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
