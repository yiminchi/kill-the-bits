# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .em import EM
from utils.reshape import reshape_weight, reshape_back_weight, reshape_activations


class PQ(EM):
    """
    Quantizes the layer M by taking into account the input activations.
    The columns are split into n_blocks blocks and a *joint* quantizer
    is learnt for all the blocks.

    Args:
        - in_features: future size(0) of the weight matrix
        - n_centroids: number of centroids per subquantizer
        - n_iter: number of k-means iterations
        - n_blocks: number of subquantizers

    Remarks:
        - For efficiency, we subsample the input activations
    """

    def __init__(self, M, n_activations=100, n_samples=1000, eps=1e-8, sample=True,
                 n_blocks=8, n_centroids=512, n_iter=20, k=3, stride=(1, 1), padding=(1, 1), groups=1):
        super(PQ, self).__init__(n_centroids, M, eps=eps)
        self.n_activations = n_activations
        self.n_samples = n_samples
        self.sample = sample
        self.n_blocks = n_blocks
        self.n_iter = n_iter
        self.k = k
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # reshape activations and weight in the case of convolutions
        self.conv = len(M.size()) == 4
        self.M = reshape_weight(M)
        # sanity check
        assert self.M.size(0) % n_blocks == 0, "n_blocks must be a multiple of in_features"
        # initialize centroids
        M_reshaped = self.sample_weights()
        self.initialize_centroids(M_reshaped)

    def _reshape_activations(self, in_activations):
        """
        Rehshapes if conv or fully-connected.
        """

        self.in_activations = reshape_activations(in_activations,
                                                  k=self.k,
                                                  stride=self.stride,
                                                  padding=self.padding,
                                                  groups=self.groups)

    def unroll_activations(self, in_activations):
        """
        Unroll activations.
        n_blocks: k * k * in_features // block_size
        in_activations: (N x H x W) x (k x k x C_in) -> (n_blocks x N x H x W) x block_size
        """

        return torch.cat(in_activations.chunk(self.n_blocks, dim=1), dim=0)

    def unroll_weight(self, M):
        """
        Unroll weights.
        n_blocks: k * k * in_features // block_size
        M: weight matrix of size (k x k x C_in) x C_out -> block_size x (n_blocks x C_out).
        """

        return torch.cat(M.chunk(self.n_blocks, dim=0), dim=1)

    def sample_activations(self):
        """
        Sample activations.
        Cuz unroll activation will make dim 0 multiply n_blocks times,
        making n_sample divide by n_blocks.
        """

        # get indices, cuz unroll activation will make dim 0 mutply n_blocks times,
        indices = torch.randint(low=0, high=self.in_activations.size(0), size=(self.n_samples // self.n_blocks,)).long()

        # sample current in_activations
        in_activations = self.unroll_activations(self.in_activations[indices])
        return in_activations.cuda()

    def sample_weights(self):
        """
        Sample weights (no sampling done here, only the unrolling).
        """

        return self.unroll_weight(self.M).cuda()

    def encode(self):
        """
        Args:
            - in_activations: input activations of size (n_samples x in_features)
            - M: weight matrix of the layer, of size (in_features x out_features)
        """

        # initialize sampling
        in_activations_reshaped_eval = self.sample_activations()
        in_activations_reshaped = self.sample_activations()
        M_reshaped = self.sample_weights()

        # perform EM training steps
        for i in range(self.n_iter):
            if self.sample:
                in_activations_reshaped = self.sample_activations()
            self.step(in_activations_reshaped, in_activations_reshaped_eval, M_reshaped, i)

    def decode(self, redo=False):
        """
        Args:
            - in_activations: input activations of size (n_samples x in_features)d
            - M: weight matrix of the layer, of size (in_features x out_features)
        """
        # use given activations to assign weightsgiven self.centroids
        if redo:
            in_activations_reshaped = self.sample_activations()
            M_reshaped = self.sample_weights()
            assignments = self.assign(in_activations_reshaped, M_reshaped)
            self.assignments = assignments
        else:
            assignments = self.assignments

        M_hat_reshaped = torch.cat(self.centroids[assignments].t().chunk(self.n_blocks, dim=1), dim=0) #(n_blocks x block_size, C_out)
        return reshape_back_weight(M_hat_reshaped, k=self.k, conv=self.conv)
