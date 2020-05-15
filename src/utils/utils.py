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
