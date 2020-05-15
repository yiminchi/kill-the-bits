# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F


def reshape_weight(weight):
    """
    C_out x C_in x k x k -> C_in x (C_out x k x k).
    """

    if len(weight.size()) == 4:
        C_out, C_in, k, k = weight.size()
        return weight.permute(1, 0, 2, 3).reshape(C_in, C_out * k * k)
    else:
        return weight.t()


def reshape_back_weight(weight, k=3, conv=True):
    """
    C_in x (C_out x k x k). -> C_out x C_in x k x k.
    """

    if conv:
        C_in, C_out_ = weight.size()
        C_out = C_out_ // (k * k)
        return weight.reshape(C_in, C_out, k, k).permute(1, 0, 2, 3)
    else:
        return weight.t()


def reshape_activations(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1):
    """
    N x C_in x H x W -> (k x k x N x H x W) x C_in.
    """
    assert groups == 1, 'Not support depthwise convolution currently'
    if len(activations.size()) == 4:
        # gather activations
        b, C_in, _, _ = activations.size()
        a_reshaped = F.unfold(activations, k, 1, padding, stride) # N x (C_in x k x k) x (H x W)
        a_reshaped = a_reshaped.reshape(b, C_in, k * k, -1).permute(2, 0, 3, 1).reshape(-1, C_in)

        return a_reshaped

    else:
        return activations
