# modify from https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation/deeplabv3.py

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import ENHANCE_MODULES
from ...utils import ConvModule
from ...weight_init import init_weights

logger = logging.getLogger()


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg,
                 mode='bilinear', align_corners=True):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(in_channels, out_channels, 1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        y = super(ASPPPooling, self).forward(x)
        return F.interpolate(y,
                             size=(int(x.size(2)), int(x.size(3))),
                             mode=self.mode,
                             align_corners=self.align_corners)


@ENHANCE_MODULES.register_module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, from_layer,
                 to_layer, mode='bilinear', align_corners=True, dropout=None,
                 conv_cfg=dict(type='Conv'), norm_cfg=None, act_cfg=None):
        super(ASPP, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

        modules = [ConvModule(in_channels,
                              out_channels,
                              1,
                              bias=False,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg)]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ConvModule(in_channels, out_channels, 3, padding=rate1, dilation=rate1,
                                  bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        modules.append(ConvModule(in_channels, out_channels, 3, padding=rate2, dilation=rate2,
                                  bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        modules.append(ConvModule(in_channels, out_channels, 3, padding=rate3, dilation=rate3,
                                  bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        modules.append(ASPPPooling(in_channels, out_channels, conv_cfg, norm_cfg, act_cfg,
                                   mode=mode, align_corners=align_corners))

        self.convs = nn.ModuleList(modules)

        self.project = ConvModule(5 * out_channels, out_channels, 1,
                                  bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.with_dropout = dropout is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(dropout)

        logger.info('ASPP init weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        if self.with_dropout:
            res = self.dropout(res)
        feats_[self.to_layer] = res
        return feats_
