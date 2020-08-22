import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from ..layers import (PixelwiseNorm, EqualizedConv2d,
    EqualizedLinear)

from ..builder import DISCRIMINATORS


# class DownsampleLayer(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.avg_pool = nn.AvgPool2d(kernel_size=2)

#     def forward(self, x):
#         return self.avg_pool(x)

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class MinibatchStddev(nn.Module):

    def __init__(self, group_size=4):
        super(MinibatchStddev, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, 1, c, h, w])
        y = y - torch.mean(y, dim=0, keepdim=True)       # [NCHW] Subtract mean over batch.
        y = torch.mean(y.pow(2.), dim=0, keepdim=True)  # [CHW]  Calc variance over batch.
        y = torch.sqrt(y + 1e-8)                         # [CHW]  Calc stddev over batch.
        y = torch.mean(y, [3, 4, 5], keepdim=True).squeeze(3)     
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, 1, h, w)   # [N1HW] Replicate over batch and pixels.
        return torch.cat([x, y], 1)                      # [N(C+1)HW] Append as new fmap.

class DiscBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation, use_wscale=True):
        super().__init__()
        layers = []
        layers.append(('conv1', EqualizedConv2d(in_channels, in_channels, use_wscale=use_wscale)))
        layers.append(('act1', activation))
        layers.append(('blur', BlurLayer()))
        layers.append(('conv2', EqualizedConv2d(in_channels, out_channels, use_wscale=use_wscale, downscale=True)))
        layers.append(('act2', activation))
        # layers.append(('downsample', DownsampleLayer()))
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.layers(x)
        return x

class DiscLastBlock(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels=1, activation, resolution=4, use_wscale=True):
        super().__init__()
        layers = []
        layers.append(('mini_std', MinibatchStddev()))
        layers.append(('conv1', EqualizedConv2d(in_channels + 1, in_channels, use_wscale=use_wscale)))
        layers.append(('act1', activation))
        layers.append(('view', View(-1)))
        layers.append(('dense', EqualizedLinear(in_channels * resolution * resolution, inter_channels, use_wscale=use_wscale)))
        layers.append(('act2', activation))
        layers.append(('dense', EqualizedLinear(inter_channels, out_channels, gain=1, use_wscale=use_wscale)))
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.layers(x)
        return x

@DISCRIMINATORS.register_module()
class ProGANDisc(nn.Module):

    def __init__(self, max_channels, resolution=1024, activation=None):
        
        def nf(depth):
            assert depth >= 0
            return min(int(8192 / (2.0 ** depth )), max_channels)

        super().__init__()
        # activation
        if activation is None:
            activation = dict(type='ReLU')
        if activation['type'] == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation['type'] == 'LeakyReLU':
            self.activation = nn.LeakyReLU(activation['alpha'], inplace=True)
        else:
            raise ValueError('Unsupported activation in mapping:', activation['type'])
        # depth
        resolution_log2 = int(math.log(resolution, 2))
        assert resolution == math.pow(2, resolution_log2)
        self.total_depth = resolution_log2 - 1
        # blocks
        blocks = []
        from_rgbs = []
        for cur_depth in range(self.total_depth, 1, -1):
            cur_channels = nf(cur_depth)
            next_channels = nf(cur_depth-1)
            from_rgbs.append(EqualizedConv2d(3, cur_channels, kernel_size=1, padding=0))
            blocks.append(DiscBlock(cur_channels, next_channels, self.activation))
        from_rgbs.append(EqualizedConv2d(3, nf(1), kernel_size=1, padding=0))
        blocks.append(DiscLastBlock(nf(1), nf(1), self.activation))
        
        self.from_rgbs = nn.ModuleList(from_rgbs)
        self.blocks = nn.ModuleList(blocks)
        self.downsample_layer = nn.AvgPool2d(2)

    def forward(self, img_in, depth, alpha):
        assert depth >= 0 and depth < self.total_depth
        if depth > 0:
            # do 2 ~ depth-1 stages
            residual = self.from_rgbs[self.total_depth - depth](self.downsample_layer(img_in))
            straight = self.from_rgbs[self.total_depth - depth - 1](img_in)
            straight = self.blocks[self.total_depth - depth - 1](straight)
            x = alpha * straight + (1 - alpha) * residual
            for cur_depth in range(self.total_depth - depth, self.total_depth-1):
                x = self.blocks[cur_depth](x)
        else:
            x = self.from_rgbs[-1](img_in)
        x = self.blocks[-1](x)
        return x
        