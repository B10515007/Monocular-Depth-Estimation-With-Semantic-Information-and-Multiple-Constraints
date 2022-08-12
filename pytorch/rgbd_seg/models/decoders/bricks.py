import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import BRICKS
from ..utils import ConvModule, build_module


@BRICKS.register_module
class JunctionBlock(nn.Module):
    """JunctionBlock

    Args:
    """

    def __init__(self, top_down, lateral, post, to_layer, fusion_method=None):
        super().__init__()
        self.from_layer = {}

        self.to_layer = to_layer
        self.top_down_cfg = copy.copy(top_down)
        lateral_ = copy.copy(lateral)
        self.fusion_method = fusion_method

        self.top_down_block = []
        if self.top_down_cfg:
            self.from_layer['top_down'] = self.top_down_cfg.pop('from_layer')
            if 'trans' in self.top_down_cfg:
                self.top_down_block.append(build_module(self.top_down_cfg['trans']))
            if 'upsample' in self.top_down_cfg:
                self.top_down_block.append(build_module(self.top_down_cfg['upsample']))
        self.top_down_block = nn.Sequential(*self.top_down_block)

        if lateral_:
            self.from_layer['lateral'] = lateral_.pop('from_layer')
            if lateral_:
                self.lateral_block = build_module(lateral_)
            else:
                self.lateral_block = nn.Sequential()
        else:
            self.lateral_block = nn.Sequential()

        if post:
            self.post_block = build_module(post)
        else:
            self.post_block = nn.Sequential()

    def forward(self, top_down=None, lateral=None):
        # print('top_down0', top_down.size())
        # print('lateral0', lateral.size())

        if lateral is not None:
            lateral = self.lateral_block(lateral)

        if top_down is not None:
            top_down = self.top_down_block(top_down)
            if self.top_down_cfg.get('adapt_upsample', False):
                top_down = F.interpolate(top_down, size=lateral.size()[-2:], mode='bilinear', align_corners=True)

        # print('top_down', top_down.size())
        # print('lateral', lateral.size())

        if top_down is not None:
            if lateral is not None:
                assert self.fusion_method in ('concat', 'add')
                if self.fusion_method == 'concat':
                    feat = torch.cat([top_down, lateral], 1)
                elif self.fusion_method == 'add':
                    feat = top_down + lateral
            else:
                assert self.fusion_method is None
                feat = top_down
        else:
            assert self.fusion_method is None
            if lateral is not None:
                feat = lateral
            else:
                raise ValueError(
                    'There is neither top down feature nor lateral feature')

        feat = self.post_block(feat)
        return feat


@BRICKS.register_module
class FusionBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self,
                 method,
                 from_layers,
                 feat_strides,
                 in_channels_list,
                 out_channels_list,
                 upsample,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu', inplace=True),
                 common_stride=4,
                 ):
        super().__init__()
        assert method in ('add', 'concat')
        self.method = method
        self.from_layers = from_layers

        assert len(in_channels_list) == len(out_channels_list)

        self.blocks = nn.ModuleList()
        for idx in range(len(from_layers)):
            in_channels = in_channels_list[idx]
            out_channels = out_channels_list[idx]
            from_layer = from_layers[idx]
            feat_stride = feat_strides[idx]
            ups_num = int(
                max(1, math.log2(feat_stride) - math.log2(common_stride)))
            head_ops = []
            for idx2 in range(ups_num):
                cur_in_channels = in_channels if idx2 == 0 else out_channels
                conv = ConvModule(
                    cur_in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                head_ops.append(conv)
                if int(feat_stride) != int(common_stride):
                    head_ops.append(build_module(upsample))
            self.blocks.append(nn.Sequential(*head_ops))

    def forward(self, feats):
        outs = []
        for idx, key in enumerate(self.from_layers):
            block = self.blocks[idx]
            feat = feats[key]
            out = block(feat)
            outs.append(out)
        if self.method == 'add':
            res = torch.stack(outs, 0).sum(0)
        else:
            res = torch.cat(outs, 1)
        return res


@BRICKS.register_module
class CollectBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self, from_layer, to_layer=None):
        super().__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

    def forward(self, feats):
        if self.to_layer is None:
            return feats[self.from_layer]
        else:
            feats[self.to_layer] = feats[self.from_layer]
