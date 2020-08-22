import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class PixelwiseNorm(nn.Module):
    # from https://zhuanlan.zhihu.com/p/56244285

    def __init__(self, eps=1e-8):
        super(PixelwiseNorm, self).__init__()
        self.eps = eps # small number for numerical stability

    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.eps).rsqrt() # [N1HW]
        return x * y

class EqualizedLinear(nn.Module):
    # from https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

    def __init__(self, in_features, out_features, gain=2**(0.5), use_wscale=True, lrmul=1, bias=True):
        super(EqualizedLinear, self).__init__()
        he_std = gain * in_features**(-0.5)
        if use_wscale:
            # equalized lr
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            # customed lr multiplier
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight_param = nn.Parameter(torch.randn(out_features, in_features) * init_std)
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
            self.b_mul = lrmul
        else:
            self.bias_param = None

    def forward(self, x):
        bias = self.bias_param
        if bias is not None:
            bias *= self.b_mul
        return F.linear(input=x,
                        weight=self.weight_param * self.w_mul,
                        bias=bias)


class Upscale2d(nn.Module):
    # a non-conv method of upscaling

    def __init__(self, factor=2, gain=1):
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        assert x.dim() == 4
        x = x * self.gain
        if factor != 1:
            shape = x.shape
            x = x.view(*shape[:3], 1, shape[3], 1).expand(-1, -1, -1, factor, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

class BlurLayer(nn.Module):

    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride
    
    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1))
        return x

class Downscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):
        super().__init__()
        self.gain = gain
        self.factor = factor
        if self.factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=self.factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        if self.blur and x.dtyle == torch.float32:
            return self.blur(x)
        x = x * self.gain
        return F.avg_pool2d(x, self.factor)


class EqualizedConv2d(nn.Module):
    # from https://zhuanlan.zhihu.com/p/56244285

    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, 
            gain=2**(0.5), use_wscale=True, lrmul=1, bias=True,
            intermediate=None, upscale=False, downscale=False):
        super(EqualizedConv2d, self).__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (kernel_size * kernel_size * in_features)**(-0.5)
        self.bias = bias
        self.stride = stride
        self.padding = padding
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight_param = nn.Parameter(torch.randn(out_features, in_features, kernel_size, kernel_size) * init_std)
        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        # e.g. blur layer
        self.intermediate = intermediate
    
    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias *= self.b_mul
        # upscale
        conv_scale = False
        if self.upscale:
            if min(x.shape[2:]) >= 64:
                w = self.weight_param * self.w_mul
                w = w.permute(1, 0, 2, 3)
                w = F.pad(w, (1, 1, 1, 1))
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
                x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1)-1)//2)
                conv_scale = True
            else:
                x = self.upscale(x)
        
        if self.downscale:
            if min(x.shape[2:]) > 128:
                w = self.weight_param * self.w_mul
                w = F.pad(w, (1, 1, 1, 1))
                w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
                x = F.conv2d(x, w, stride=2, padding=(w.size(-1)-1)//2)
                conv_scale = True
            else:
                assert self.intermediate is None:
                self.intermediate = self.downscale
        # conv
        if not conv_scale:
            if not self.intermediate:
                return F.conv2d(x, weight=self.weight_param * self.w_mul, bias=bias,
                    stride=self.stride, padding=self.padding)
            else:
                x = F.conv2d(x, weight=self.weight_param * self.w_mul, bias=None,
                    stride=self.stride, padding=self.padding)
        # intermediate
        if self.intermediate:
            x = self.intermediate(x)
        if bias:
            x = x + bias.view(1, -1, 1, 1)
        return x


