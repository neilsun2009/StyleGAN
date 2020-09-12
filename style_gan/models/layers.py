import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

# class PixelwiseNorm(nn.Module):
#     # from https://zhuanlan.zhihu.com/p/56244285

#     def __init__(self, eps=1e-8):
#         super(PixelwiseNorm, self).__init__()
#         self.eps = eps # small number for numerical stability

#     def forward(self, x):
#         y = x.pow(2.).mean(dim=1, keepdim=True).add(self.eps).rsqrt() # [N1HW]
#         return x * y

class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
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
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x

class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return F.avg_pool2d(x, self.factor)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class EqualizedConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5, use_wscale=False,
                 lrmul=1, bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x

# class EqualizedLinear(nn.Module):
#     # from https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

#     def __init__(self, in_features, out_features, gain=2**(0.5), use_wscale=True, lrmul=1, bias=True):
#         super(EqualizedLinear, self).__init__()
#         he_std = gain * in_features**(-0.5)
#         if use_wscale:
#             # equalized lr
#             init_std = 1.0 / lrmul
#             self.w_mul = he_std * lrmul
#         else:
#             # customed lr multiplier
#             init_std = he_std / lrmul
#             self.w_mul = lrmul
#         self.weight_param = nn.Parameter(torch.randn(out_features, in_features) * init_std)
#         if bias:
#             self.bias_param = nn.Parameter(torch.zeros(out_features))
#             self.b_mul = lrmul
#         else:
#             self.bias_param = None

#     def forward(self, x):
#         bias = self.bias_param
#         if bias is not None:
#             bias = bias * self.b_mul
#         return F.linear(input=x,
#                         weight=self.weight_param * self.w_mul,
#                         bias=bias)


# class Upscale2d(nn.Module):
#     # a non-conv method of upscaling

#     def __init__(self, factor=2, gain=1):
#         super().__init__()
#         self.gain = gain
#         self.factor = factor

#     def forward(self, x):
#         assert x.dim() == 4
#         x = x * self.gain
#         if self.factor != 1:
#             shape = x.shape
#             x = x.view(*shape[:3], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
#             x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
#         return x

# class BlurLayer(nn.Module):

#     def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
#         super().__init__()
#         kernel = torch.tensor(kernel, dtype=torch.float32)
#         kernel = kernel[:, None] * kernel[None, :]
#         kernel = kernel[None, None]
#         if normalize:
#             kernel = kernel / kernel.sum()
#         if flip:
#             kernel = kernel[:, :, ::-1, ::-1]
#         self.register_buffer('kernel', kernel)
#         self.stride = stride
    
#     def forward(self, x):
#         kernel = self.kernel.expand(x.size(1), -1, -1, -1)
#         x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2),
#             groups=x.size(1))
#         return x

# class Downscale2d(nn.Module):

#     def __init__(self, factor=2, gain=1):
#         super().__init__()
#         self.gain = gain
#         self.factor = factor
#         if self.factor == 2:
#             f = [np.sqrt(gain) / self.factor] * self.factor
#             self.blur = BlurLayer(kernel=f, normalize=False, stride=self.factor)
#         else:
#             self.blur = None

#     def forward(self, x):
#         assert x.dim() == 4
#         if self.blur and x.dtype == torch.float32:
#             return self.blur(x)
#         x = x * self.gain
#         return F.avg_pool2d(x, self.factor)


# class EqualizedConv2d(nn.Module):
#     # from https://zhuanlan.zhihu.com/p/56244285

#     def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, 
#             gain=2**(0.5), use_wscale=True, lrmul=1, bias=True,
#             intermediate=None, upscale=False, downscale=False):
#         super(EqualizedConv2d, self).__init__()
#         if upscale:
#             self.upscale = Upscale2d()
#         else:
#             self.upscale = None
#         if downscale:
#             self.downscale = Downscale2d()
#         else:
#             self.downscale = None
#         he_std = gain * (kernel_size * kernel_size * in_features)**(-0.5)
#         self.bias = bias
#         self.stride = stride
#         self.padding = padding
#         if use_wscale:
#             init_std = 1.0 / lrmul
#             self.w_mul = he_std * lrmul
#         else:
#             init_std = he_std / lrmul
#             self.w_mul = lrmul
#         self.weight_param = nn.Parameter(torch.randn(out_features, in_features, kernel_size, kernel_size) * init_std)
#         if self.bias:
#             self.bias_param = nn.Parameter(torch.zeros(out_features))
#             self.b_mul = lrmul
#         else:
#             self.bias_param = None
#         # e.g. blur layer
#         self.intermediate = intermediate
    
#     def forward(self, x):
#         bias = self.bias_param
#         if bias is not None:
#             bias = bias * self.b_mul
#         # upscale
#         conv_scale = False
#         if self.upscale is not None:
#             if min(x.shape[2:]) >= 64:
#                 w = self.weight_param * self.w_mul
#                 w = w.permute(1, 0, 2, 3)
#                 w = F.pad(w, (1, 1, 1, 1))
#                 w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
#                 x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1)-1)//2)
#                 conv_scale = True
#             else:
#                 x = self.upscale(x)
        
#         downscale = self.downscale
#         intermediate = self.intermediate
#         if downscale is not None:
#             if min(x.shape[2:]) >= 128:
#                 w = self.weight_param * self.w_mul
#                 w = F.pad(w, (1, 1, 1, 1))
#                 w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
#                 x = F.conv2d(x, w, stride=2, padding=(w.size(-1)-1)//2)
#                 conv_scale = True
#             else:
#                 assert intermediate is None
#                 intermediate = downscale
#         # conv
#         if not conv_scale:
#             if intermediate is None:
#                 return F.conv2d(x, weight=self.weight_param * self.w_mul, bias=bias,
#                     stride=self.stride, padding=self.padding)
#             else:
#                 x = F.conv2d(x, weight=self.weight_param * self.w_mul, bias=None,
#                     stride=self.stride, padding=self.padding)
#         # intermediate
#         if intermediate is not None:
#             x = intermediate(x)
#         if bias is not None:
#             x = x + bias.view(1, -1, 1, 1)
#         return x


