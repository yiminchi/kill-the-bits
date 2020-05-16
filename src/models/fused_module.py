import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.utils import *

class _LQ(nn.Module):
    def __init__(self):
        super(_LQ, self).__init__()
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.quantize = True

    def quantize_bias(self, b):
        """
        Symmetric Linear Quantization with 16 bits
        make zero point : 2**15
        S7.8
        """

        if self.quantize:
            b = LinearQuantizeSTE.apply(b, 1/2**8, 2**15, 16, False)
        return b
    
    def quantize_weight(self, w):
        """
        Asymmetric Linear Quantization with 8 bits
        """

        with torch.no_grad():   quant_weight_param(w, self.scale, self.zero_point)
        if self.quantize:
            w = LinearQuantizeSTE.apply(w, self.scale, self.zero_point, 8, False)
        return w

class conv_LQ(_LQ):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        super(conv_LQ, self).__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        w_quantized, bias_quantized = self.quantize_weight(self.weight), self.quantize_bias(self.bias)
        return F.conv2d(x, w_quantized, bias_quantized, self.stride,
                    self.padding, self.dilation, self.groups)

class linear_LQ(_LQ):
    def __init__(self, weight, bias):
        super(linear_LQ, self).__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        w_quantized, bias_quantized = self.quantize_weight(self.weight), self.quantize_bias(self.bias)
        return F.linear(x, w_quantized, bias_quantized)

class IO_LQ(nn.Module):
    def __init__(self):
        super(IO_LQ, self).__init__()
        """
        Signed asymmetric linear quantizatize on inout port
        scale can only be power of 2 for the hardware implement
        """

        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        self.num_batch = 0
        self.calibration = False
        self.quantize = True

    def forward(self, input):
        return self.quantize_act(input)

    def calibrate(self, input):
        """
        Using exponential moving averages to calibrate the scale
        """
        param_min, param_max = get_tensor_min_max(input)
        self.scale[:] = ((param_max-param_min)/(2**8 - 1) + self.num_batch * self.scale) / (self.num_batch + 1)
        running_scale = torch.pow(2, torch.ceil(torch.log2(self.scale)))
        self.num_batch += 1
        self.zero_point[:] = torch.floor(-param_min/running_scale + 0.5)

    def quantize_act(self, input):
        if self.training:
            assert not self.calibration # cannot calibrate when training
            with torch.no_grad(): 
                running_scale = quant_act_param(input, self.scale, self.zero_point)
        else:
            if self.calibration: self.calibrate(input)
            running_scale = torch.pow(2, torch.ceil(torch.log2(self.scale)))

        if self.quantize:
            input = LinearQuantizeSTE.apply(input, running_scale, self.zero_point, 8, False)
        return input

class ReLU_LQ(IO_LQ):
    def __init__(self):
        super(ReLU_LQ, self).__init__()

    def forward(self, input):
        input = input.clamp_min_(0)
        return self.quantize_act(input)