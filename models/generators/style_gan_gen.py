import mmcv
import numpy as np
import torch
import torch.nn as nn

from ..builder import (build_gen_downsampling,
    build_gen_upsampling, build_gen_mapping,
    build_gen_synthesis,
    GENERATORS)

@GENERATORS.register_module()
class StyleGANGen(nn.Module):

    # Todo: trunction trick

    def __init__(self, mapping, synthesis, resolution):
        super().__init__()
        self.mapping = build_gen_mapping(mapping, resolution=resolution)
        self.synthesis = build_gen_synthesis(synthesis, resolution=resolution)
    
    def forward(self, input_latent, depth, alpha):
        style_latent = self.mapping(input_latent)
        img_out = self.synthesis(style_latent, depth, alpha)
        return img_out