import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class PixelwiseNorm(nn.Module):
    # from https://zhuanlan.zhihu.com/p/56244285

    def __init__(self, eps=1e-8):
        super(PixelwiseNorm, self).__init__()
        self.eps = eps # small number for numerical stability

    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.eps).rsqrt() # [N1HW]
        return x * y

class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, bias_init=0):
        super(EqualizedLinear, self).__init__()
        self.bias = bias
        self.weight_param = nn.Parameter(torch.FloatTensor(out_features, in_features).normal_(0.0, 1.0))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_features).fill_(bias_init))
        fan_in = in_features
        self.scale = math.sqrt(2. / fan_in)

    def forward(self, x):
        return F.linear(input=x,
                        weight=self.weight_param.mul(self.scale),
                        bias=self.bias_param if self.bias else None)

class EqualizedConv2d(nn.Module):
    # from https://zhuanlan.zhihu.com/p/56244285

    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, bias=True):
        super(EqualizedConv2d, self).__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.weight_param = nn.Parameter(torch.FloatTensor(out_features, in_features, kernel_size, kernel_size).normal_(0.0, 1.0))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_features).fill_(0))
        fan_in = kernel_size * kernel_size * in_features
        self.scale = math.sqrt(2. / fan_in)
    
    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.weight_param.mul(self.scale),  # scale the weight on runtime
                        bias=self.bias_param if self.bias else None,
                        stride=self.stride, padding=self.padding)

