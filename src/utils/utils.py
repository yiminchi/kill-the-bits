# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from .reshape import reshape_weight, reshape_back_weight


def centroids_from_weights(M, assignments, n_centroids, n_blocks):
    """
    Recovers the centroids from an already quantized matrix M.

    Args:
        - M: already quantized matrix
        - assignments: size (n_blocks, n_vectors)
        - n_centroids: number of centroids
        - n_blocks: niumber of blocks per column

    Remarks:
        - This function assumes that all the clusters are non-empty
        - This function consists in two steps:
            (1) reshape the 2D/4D matrix (whether fully-connected or convolutional) into a 2D matrix
            (2) unroll the obtained 2D matrix according to the number of blocks
    """

    M_reshaped = reshape_weight(M)
    M_unrolled = torch.stack(M_reshaped.chunk(n_blocks, dim=0), dim=0)
    size_block = M_unrolled.size(1)
    centroids = torch.zeros(n_blocks, n_centroids, size_block, device=M.device)
    for m in range(n_blocks):
        for k in range(n_centroids):
            centroids[m, k] = M_unrolled[m, :, assignments[m] == k][:, 0]

    return centroids


def weight_from_centroids(centroids, assignments, n_blocks, k, conv):
    """
    Constructs the 2D matrix from its centroids.

    Args:
        - centroids: size (n_blocks, block_size x n_centroids)
        - assignments: size (n_blocks, n_vectors)
        _ n_blocks: numnber of blocks per column
        - k: kernel size (set to 1 if not is_conv)
        - is_conv: convolutional or linear layer

    Remarks:
        - This function consists in two steps:
            (1) get the 2D unrolled weight matrix
            (2) reshape it in the case of fully-connected of convolutional layer
    """

    M_hat_reshaped = torch.gather(centroids, 1, assignments.unsqueeze(2).repeat(1, 1, centroids.size(2))) #(n_blocks, C_out x k x k, block_size)
    M_hat_reshaped = M_hat_reshaped.permute(0, 2, 1)
    M_hat_reshaped = M_hat_reshaped.reshape(-1, M_hat_reshaped.size(2)) #(C_in,  C_out x k x k)
    return reshape_back_weight(M_hat_reshaped, k=k, conv=conv)

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits=8, inplace=True):
        if inplace:
            ctx.mark_dirty(input)
        input = quantize(input, scale, zero_point, num_bits=num_bits, inplace=inplace)
        input = dequantize(input, scale, zero_point, inplace=inplace)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

def quant_weight_param(param, scale, zero_point, num_bits=8):
    """
    Asymmetric
    """
    param_min, param_max = get_tensor_min_max(param)
    scale[:] = (param_max-param_min)/(2**num_bits - 1)
    scale = quantize(scale, 1/2**16, 0, inplace=True, num_bits=16)
    scale.clamp_min_(1)  # avoid the zero scale
    scale = dequantize(scale, 1/2**16, 0, inplace=True)
    zero_point[:] = torch.floor(-param_min/scale + 0.5)

def quant_act_param(param, scale, zero_point, num_bits=8):
    """
    Asymmetric
    """
    param_min, param_max = get_tensor_min_max(param)
    scale[:] = 0.9 * (param_max-param_min)/(2**num_bits - 1) + 0.1 * scale
    running_scale = torch.pow(2, torch.ceil(torch.log2(scale)))
    zero_point[:] = torch.floor(-param_min/running_scale + 0.5)
    return running_scale

def get_tensor_min_max(t):
    param_min, param_max = t.min(), t.max()
    return torch.clamp_max(param_min, 0), torch.clamp_min(param_max, 0)

def quantize(input, scale, zero_point, inplace=True, num_bits=8):
    if inplace:
        input.div_(scale).add_(0.5).floor_().add_(zero_point).clamp_(0, 2 ** num_bits - 1)
        return input
    return torch.clamp(torch.floor(input / scale + 0.5) + zero_point, 0, 2 ** num_bits - 1)

def dequantize(input, scale, zero_point, inplace=True):
    if inplace:
        input.add_(0.5).floor_().sub_(zero_point).mul_(scale)
        return input
    return (torch.floor(input + 0.5) - zero_point) * scale