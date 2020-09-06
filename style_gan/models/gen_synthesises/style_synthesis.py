import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from ..layers import (PixelwiseNorm, EqualizedConv2d,
    EqualizedLinear, BlurLayer)

from ..builder import (GEN_SYNTHESISES)

class InputLayer(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.const = nn.Parameter(torch.ones(1, out_channels, 4, 4))
        # need bias?
        self.bias = nn.Parameter(torch.ones(out_channels))

    def forward(self, style_latent):
        batch_size = style_latent.size(0)
        x = self.const.expand(batch_size, -1, -1, -1)
        return x + self.bias.view(1, -1, 1, 1)

# class UpsampleLayer(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return F.interpolate(x, scale_factor=2, mode='nearest')

class NoiseLayer(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x

class StyleAdaIN(nn.Module):

    def __init__(self, channels, style_channels, use_wscale=True):
        super().__init__()
        # TODO instance norm place
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.affine = EqualizedLinear(style_channels, channels * 2,
            gain=1.0, use_wscale=use_wscale)

    def forward(self, x, style_latent):
        x = self.instance_norm(x)
        style = self.affine(style_latent)
        shape = [-1, 2, x.size(1)] + [1] * (x.dim() -2)
        style = style.view(shape)
        x = x * (style[:, 0] + 1.0) + style[:, 1]
        return x

class PostConv(nn.Module):

    def __init__(self, channels, style_channels, activation, use_wscale=True):
        super().__init__()
        self.noise_layer = NoiseLayer(channels)
        self.activation = activation
        self.style_adain = StyleAdaIN(channels, style_channels, use_wscale=use_wscale)

    def forward(self, x, style_latent):
        x = self.noise_layer(x)
        x = self.activation(x)
        x = self.style_adain(x, style_latent)
        return x

class SynthesisBlock(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels,
            activation, is_first_block=False, use_wscale=True, blur_filter=None):
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.is_first_block = is_first_block
        if self.is_first_block:
            self.input_layer = InputLayer(out_channels)
        else:
            self.conv1 = EqualizedConv2d(in_channels, out_channels, use_wscale=use_wscale,
                upscale=True, intermediate=blur)
        self.post_conv1 = PostConv(out_channels, style_channels,
            activation, use_wscale=use_wscale)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, use_wscale=use_wscale)
        self.post_conv2 = PostConv(out_channels, style_channels,
            activation, use_wscale=use_wscale)
    
    def forward(self, x, style_latent):
        if self.is_first_block:
            x = self.input_layer(style_latent)
        else:
            x = self.conv1(x)
        x = self.post_conv1(x, style_latent[:, 0])
        x = self.conv2(x)
        x = self.post_conv2(x, style_latent[:, 1])
        return x


@GEN_SYNTHESISES.register_module()
class StyleSynthesis(nn.Module):
    # ref https://github.com/SaoYan/GenerativeSkinLesion/blob/master/networks.py

    def __init__(self, in_channels, style_channels=512,
            resolution=1024, activation=None, use_wscale=True, blur_filter=[1, 2, 1]):

        def nf(depth):
            assert depth >= 0
            if depth == 0:
                return in_channels
            return min(int(8192 / (2.0 ** depth )), 512)

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
        # later depth no starts from 0
        # blocks
        blocks = [SynthesisBlock(nf(0), nf(1),
            style_channels, self.activation, is_first_block=True, use_wscale=True)]
        to_rgbs = [EqualizedConv2d(nf(1), 3, kernel_size=1, padding=0, gain=1, use_wscale=True)]
        for depth in range(2, self.total_depth + 2):
            last_channels = nf(depth-1)
            cur_channels = nf(depth)
            blocks.append(SynthesisBlock(last_channels, cur_channels,
                style_channels, self.activation, use_wscale=True, blur_filter=blur_filter))
            to_rgbs.append(EqualizedConv2d(cur_channels, 3, kernel_size=1, padding=0, gain=1, use_wscale=True))
        self.blocks = nn.ModuleList(blocks)
        self.to_rgbs = nn.ModuleList(to_rgbs)
        self.upsample_layer = lambda x: F.interpolate(x, scale_factor=2)

    def forward(self, style_latent, depth, alpha):
        assert depth >= 0 and depth < self.total_depth
        x = self.blocks[0](None, style_latent[:, :2])
        if depth > 0:
            # do 2 ~ depth-1 stages
            for cur_depth in range(1, depth):
                x = self.blocks[cur_depth](x, style_latent[:, 2*cur_depth:2*cur_depth+2])
            residual = self.to_rgbs[depth - 1](self.upsample_layer(x))
            # do depth stage
            x = self.blocks[depth](x, style_latent[:, 2*depth:2*depth+2])
            straight = self.to_rgbs[depth](x)
            img_out = alpha * straight + (1 - alpha) * residual
        else:
            img_out = self.to_rgbs[0](x)
        return img_out