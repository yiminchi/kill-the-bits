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
    C_out x C_in x k x k -> (k x k x C_in) x C_out.
    """

    if len(weight.size()) == 4:
        C_out, C_in, k, k = weight.size()
        return weight.permute(2, 3, 1, 0).reshape(k * k * C_in, C_out)
    else:
        return weight.t()


def reshape_back_weight(weight, k=3, conv=True):
    """
    (k x k x C_in) x C_out. -> C_out x C_in x k x k.
    """

    if conv:
        C_in_, C_out = weight.size()
        C_in = C_in_ // (k * k)
        return weight.reshape(k, k, C_in, C_out).permute(3, 2, 0, 1)
    else:
        return weight.t()


def reshape_activations(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1):
    """
    N x C_in x H x W -> (N x H x W) x (k x k x C_in).
    """
    assert groups == 1, 'Not support depthwise convolution currently'
    if len(activations.size()) == 4:
        # gather activations
        b, C_in, _, _ = activations.size()
        a_reshaped = F.unfold(activations, k, 1, padding, stride) # N x (C_in x k x k) x (H x W)
        a_reshaped = a_reshaped.reshape(b, C_in, k * k, -1).permute(0, 3, 2, 1).reshape(-1, k * k * C_in)

        return a_reshaped

    else:
        return activations
