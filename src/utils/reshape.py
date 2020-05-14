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
    C_out x C_in x k x k -> (C_in x k x k) x C_out.
    """

    if len(weight.size()) == 4:
        C_out, C_in, k, k = weight.size()
        return weight.view(C_out, C_in * k * k).t()
    else:
        return weight.t()


def reshape_back_weight(weight, k=3, conv=True):
    """
    (C_in x k x k) x C_out -> C_out x C_in x k x k.
    """

    if conv:
        C_in_, C_out = weight.size()
        C_in = C_in_ // (k * k)
        return weight.t().view(C_out, C_in, k, k)
    else:
        return weight.t()


def reshape_activations(activations, k=3, stride=(1, 1), padding=(1, 1), groups=1):
    """
    N x C_in x H x W -> (N x H x W) x (C_in x k x k).
    """

    if len(activations.size()) == 4:
        # gather activations
        b, C_in, _, _ = activations.size()
        a_reshaped = F.unfold(activations, k, 1, padding, stride) # N x (C_in x k x k) x (H x W)
        a_reshaped = a_reshaped.permute(0, 2, 1).reshape(-1, C_in * k * k)

        # group convolutions (e.g. depthwise convolutions)
        a_reshaped_groups = torch.cat(a_reshaped.chunk(groups, dim=1), dim=0)

        return a_reshaped_groups

    else:
        return activations
