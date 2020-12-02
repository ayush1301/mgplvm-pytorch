from __future__ import print_function
import numpy as np
from mgplvm.utils import softplus
from . import sgp, svgp
from .. import rdist, kernels, utils
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import pickle
import mgplvm.lpriors as lpriors


class SvgpLvm(nn.Module):
    name = "Svgplvm"

    def __init__(self,
                 n,
                 z,
                 kernel,
                 likelihood,
                 lat_dist,
                 lprior,
                 whiten=True):
        """
        __init__ method for Vanilla model
        Parameters
        ----------
        n : int
            number of neurons
        m : int
            number of conditions
        z : Inducing Points
            inducing points
        kernel : Kernel
            kernel used for GP regression
        likelihood : Likelihood
            likelihood p(y|f)
        lat_dist : rdist
            latent distribution
        lprior: Lprior
            log prior over the latents
        whiten: bool
            parameter passed to Svgp
        """
        super().__init__()
        self.n = n
        self.kernel = kernel
        self.z = z
        self.likelihood = likelihood
        self.whiten = whiten
        self.svgp = svgp.Svgp(self.kernel,
                              n,
                              self.z,
                              likelihood,
                              whiten=whiten)
        # latent distribution
        self.lat_dist = lat_dist
        self.lprior = lprior

    def forward(self, data, n_mc, kmax=5, batch_idxs=None):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n x m x n_samples)
        n_mc : int
            number of MC samples
        kmax : int
            parameter for estimating entropy for several manifolds
            (not used for some manifolds)
        batch_idxs: Optional int list
            if None then use all data and (batch_size == m)
            otherwise, (batch_size == len(batch_idxs))
        Returns
        -------
        svgp_elbo : Tensor
            evidence lower bound of sparse GP per batch
        kl : Tensor
            estimated KL divergence per batch between variational distribution 
            and a manifold-specific prior (uniform for all manifolds except 
            Euclidean, for which we have a Gaussian N(0,I) prior)
        Notes
        ----
        ELBO of the model per batch is [ svgp_elbo - kl ]
        """

        data = data if batch_idxs is None else data[:, batch_idxs, :]
        _, _, n_samples = data.shape
        g, lq = self.lat_dist.sample(torch.Size([n_mc]), batch_idxs)
        # sparse GP elbo summed over all batches
        # note that [ svgp.elbo ] recognizes inputs of dims (n_mc x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        svgp_lik, svgp_kl = self.svgp.elbo(n_samples, n_mc, data,
                                           g.permute(0, 2, 1))
        # KL(Q(G) || p(G)) ~ logQ - logp(G)
        kl = lq.sum() - self.lprior(g).sum()
        return svgp_lik / n_mc, svgp_kl / n_mc, (kl / n_mc)