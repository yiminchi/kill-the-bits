import torch
import torch.nn.functional as F
from operator import attrgetter

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

def _convert_model(module, prev_name, layer, type):
    for name, child in module.named_children():
        if list(child.named_children()):
            _convert_model(child, prev_name+name+'.', layer, type)

        for name, mod in list(module.named_children()):
            if prev_name+name == layer:
                setattr(module, name, type())
                print('Convert {} to'.format(layer), type())
                return

def fuse_ConvBn(model, conv, bn):
    conv_layer = attrgetter(conv)(model)
    bn_layer = attrgetter(bn)(model)
    weight, bias = fuse_convbn_param(conv_layer.weight, conv_layer.bias, bn_layer.weight, bn_layer.bias, 
                                bn_layer.running_var, bn_layer.running_mean, bn_layer.eps)
    conv_layer.weight = torch.nn.Parameter(weight)
    conv_layer.bias = torch.nn.Parameter(bias)
    _convert_model(model, '', bn, type=torch.nn.Identity)