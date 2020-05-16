import torch
import torch.nn.functional as F
from operator import attrgetter
from models.fused_module import *

def broadcast_correction_weight(c: torch.Tensor):
    """
    Broadcasts a correction factor to the weight.
    """
    if c.dim() != 1:
        raise ValueError("Correction factor needs to have a single dimension")
    expected_weight_dim = 4
    view_fillers_dim = expected_weight_dim - c.dim()
    view_filler = (1,) * view_fillers_dim
    expected_view_shape = c.shape + view_filler
    return c.view(*expected_view_shape)

def fuse_convbn_param(weight, bias, gamma, beta, running_var, running_mean, eps):
    recip_sigma_running = torch.rsqrt(running_var + eps)
    weight_corrected = weight * broadcast_correction_weight(gamma * recip_sigma_running)

    corrected_mean = running_mean - (bias if bias is not None else 0)
    bias_corrected = beta - gamma * corrected_mean * recip_sigma_running

    return weight_corrected, bias_corrected

def _convert_model(module, prev_name, layer, replace):
    for name, child in module.named_children():
        if list(child.named_children()):
            _convert_model(child, prev_name+name+'.', layer, replace)

        for name, mod in list(module.named_children()):
            if prev_name+name == layer:
                setattr(module, name, replace)
                print('Convert {} to'.format(layer), replace)
                return

def fuse_ConvBn(model, conv, bn):
    conv_layer = attrgetter(conv)(model)
    bn_layer = attrgetter(bn)(model)
    weight, bias = fuse_convbn_param(conv_layer.weight, conv_layer.bias, bn_layer.weight, bn_layer.bias, 
                                bn_layer.running_var, bn_layer.running_mean, bn_layer.eps)
    replace_conv = conv_LQ(weight, bias, conv_layer.stride, conv_layer.padding, conv_layer.dilation, conv_layer.groups)
    _convert_model(model, '', conv, replace=replace_conv)
    _convert_model(model, '', bn, replace=torch.nn.Identity())
    return model.cuda()

def replace_relu(model, relu):
    _convert_model(model, '', relu, replace=ReLU_LQ())
    return model.cuda()

def replace_linear(model, linear):
    linear_layer = attrgetter(linear)(model)
    _convert_model(model, '', linear, replace=linear_LQ(linear_layer.weight, linear_layer.bias))
    return model.cuda()

def insert_io(model, input=True):
    if input:
        model = nn.Sequential(IO_LQ(), model)
    else:
        model.add_module('last', IO_LQ())
    return model.cuda()