# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class SECOND3DFPN(BaseModule):
    """Modified FPN used in SECOND.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        use_for_distill (bool): Whether to use for cross-modality distillation.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv3d', bias=False),
                 conv_cfg=dict(type='Conv3d', bias=False),
                 extra_conv=None,
                 use_conv_for_no_stride=False,
                 use_for_distill=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECOND3DFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extra_conv = extra_conv
        self.fp16_enabled = False
        self.use_for_distill = use_for_distill

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=(1,stride,stride) if '3d' in upsample_cfg['type'] else (stride,stride),
                    stride=(1,stride,stride) if '3d' in upsample_cfg['type'] else (stride,stride))
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=(1,stride,stride) if '3d' in conv_cfg['type'] else (stride,stride),
                    stride=(1,stride,stride) if '3d' in conv_cfg['type'] else (stride,stride))

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if self.extra_conv is not None:
            extra_blocks = []
            self.layer_num = self.extra_conv.pop('num_conv')
            if "kernel" in self.extra_conv:
                kernel = self.extra_conv.pop("kernel")
            else:
                kernel = (3,3,3)
            padding = tuple([(_k-1)//2 for _k in kernel])
            if "sep_kernel" in self.extra_conv:
                sep_kernel = self.extra_conv.pop("sep_kernel")
                sep_padding = tuple([(_k-1)//2 for _k in sep_kernel])
            else:
                sep_kernel = None
            for j in range(self.layer_num):
                extra_blocks.append(
                    build_conv_layer(
                        self.extra_conv,
                        out_channels[-1],
                        out_channels[-1],
                        kernel,
                        padding=padding))
                if sep_kernel:
                    extra_blocks.append(
                        build_conv_layer(
                            self.extra_conv,
                            out_channels[-1],
                            out_channels[-1],
                            sep_kernel,
                            padding=sep_padding))
                extra_blocks.append(build_norm_layer(norm_cfg, out_channels[-1])[1])
                extra_blocks.append(nn.ReLU(inplace=True))
            self.extra_blocks = nn.Sequential(*extra_blocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = sum(ups)
        else:
            out = ups[0]

        if self.extra_conv is not None:
            if self.use_for_distill:
                out_final = out
                before_relu_list = []
                for _idx in range(self.layer_num):
                    out_mid = self.extra_blocks[_idx*3:(_idx+1)*3-1](out_final)
                    out_before_relu = out_mid.clone()
                    out_final = self.extra_blocks[(_idx+1)*3-1](out_mid)
                    before_relu_list.append(out_before_relu)
                
                out = {'final':out_final, 'before_relu':before_relu_list}
            else:
                out = self.extra_blocks(out)
        return out