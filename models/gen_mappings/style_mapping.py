import mmcv
import numpy as np
import torch
import torch.nn as nn
import math
from collections import OrderedDict
from ..layers import (PixelwiseNorm, EqualizedLinear)

from ..builder import (GEN_MAPPINGS)

@GEN_MAPPINGS.register_module()
class StyleMapping(nn.Module):

    def __init__(self, in_channels=512, out_channels=512, num_layers=8,
            activation=None, resolution=1024):
        super().__init__()
        self.num_layers = num_layers
        resolution_log2 = int(math.log(resolution, 2))
        assert resolution == math.pow(2, resolution_log2)
        self.broadcast_num = (resolution_log2 - 1) * 2
        if activation is None:
            activation = dict(type='ReLU')
        if activation['type'] == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation['type'] == 'LeakyReLU':
            self.activation = nn.LeakyReLU(activation['alpha'], inplace=True)
        else:
            raise ValueError('Unsupported activation in mapping:', activation['type'])
        layers = list()
        layers.append(('mapping_norm', PixelwiseNorm()))
        layers.append(('mapping_dense_0', nn.EqualizedLinear(in_channels, out_channels)))
        layers.append(('mapping_act_0', self.activation))
        for layer_idx in range(1, num_layers):
            layers.append(('mapping_dense_{}'.format(layer_idx),
                nn.EqualizedLinear(out_channels, out_channels)))
            layers.append(('mapping_act_{}'.format(layer_idx), self.activation))
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.layers(x)
        x = x.unsqueeze(1).view(-1, self.broadcast_num, -1)
        return x
