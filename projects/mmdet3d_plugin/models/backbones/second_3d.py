# Copyright (c) OpenMMLab. All rights reserved.
from symbol import import_from
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES


@BACKBONES.register_module()
class SECOND3D(BaseModule):
    """Modified Backbone network for SECOND.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 is_cascade=True,
                 norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv3d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SECOND3D, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        
        if isinstance(in_channels, list):
            in_filters = in_channels
        else:
            in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        self.is_cascade = is_cascade
        self.kernel_type = conv_cfg.type
        if "kernel" in conv_cfg:
            kernel = conv_cfg.pop("kernel")
        else:
            kernel = (1,3,3)
        padding = tuple([(_kernel-1)//2 for _kernel in kernel])
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    kernel,
                    stride=(1,layer_strides[i],layer_strides[i]) if len(padding)==3 else (layer_strides[i],layer_strides[i]),
                    padding=padding),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        kernel,
                        padding=padding))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer=self.kernel_type)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        batch = x.shape[0]
        if self.kernel_type == "Conv2d":
            x = x.transpose(1,2).flatten(0,1)

        for i in range(len(self.blocks)):
            if self.is_cascade:
                x = self.blocks[i](x)
                outs.append(x)
            else:
                out = self.blocks[i](x)
                outs.append(out)
        
        if self.kernel_type == "Conv2d":
            outs = [_out.reshape(batch, -1, *_out.shape[-3:]).transpose(1,2) for _out in outs]
        
        return tuple(outs)
