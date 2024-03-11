import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import abc
from torch.distributions import MultivariateNormal, Poisson, NegativeBinomial, Normal
from itertools import chain
import torch.distributions as dists
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch.optim.lr_scheduler import StepLR, LambdaLR
from main import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Preprocessor(Module):
    def __init__(self, v: Tensor, z_dim: int, noise_scale=1., W=None, R_half=None) -> None:
        # Generative parameters
        # z_t = A z_{t-1} + B w_t
        # v_t = W z_t + varepsilon_t

        super(Preprocessor, self).__init__()
        self.v = v.to(device) # (ntrials, v_dim, T)
        self.T = v.shape[-1]
        self.v_dim = v.shape[1]
        self.z_dim = z_dim # dimension of the latent space

        self.A = torch.nn.Parameter(0.8* torch.randn(self.z_dim, self.z_dim).to(device) / np.sqrt(self.z_dim))
        # self.pre_A_eig = torch.nn.Parameter(torch.randn(self.z_dim).to(device))
        # self._S = torch.nn.Parameter(torch.randn(self.z_dim, self.z_dim).to(device) / np.sqrt(self.z_dim))
        self.B = torch.nn.Parameter(noise_scale * torch.eye(self.z_dim).to(device))
        if W is not None:
            self.W = torch.nn.Parameter(W.to(device))
        else:
            self.W = torch.nn.Parameter(torch.randn(self.v_dim, self.z_dim).to(device) / np.sqrt(self.z_dim))
        # self.mu0 = torch.nn.Parameter(torch.rand(self.z_dim).to(device))
        self.mu0 = torch.nn.Parameter(torch.zeros(self.z_dim).to(device))
        self.Sigma0_half = torch.nn.Parameter(noise_scale * torch.eye(self.z_dim).to(device))
        # self.log_sigma_v = torch.nn.Parameter(torch.log(torch.abs(torch.randn(1).to(device))))
        if R_half is not None:
            self.R_half = torch.nn.Parameter(R_half.to(device))
        else:
            self.R_half = torch.nn.Parameter(noise_scale/10 * torch.eye(self.v_dim).to(device))

    # @property
    # def sigma_v(self):
    #     return torch.exp(self.log_sigma_v)
    
    # @property
    # def var_v(self):
    #     return torch.square(self.sigma_v)
    
    # @property
    # def R(self):
    #     jitter = torch.eye(self.v_dim).to(self.sigma_v.device) * 1e-6
    #     return self.var_v * torch.eye(self.v_dim).to(device) + jitter
    
    # @property
    # def A(self):
    #     l =  torch.diag(torch.tanh(self.pre_A_eig))
    #     S = torch.tril(self._S)  - torch.tril(self._S).T
    #     Q = torch.exp(S) # (z_dim, z_dim)
    #     A =  Q @ l @ Q.inverse()
    #     print(torch.max(torch.abs(torch.linalg.eigvals(A))))
    #     return A


    def reg(self, x):
        # Compute the exponential of the diagonal elements and add a small constant
        xd = torch.exp(torch.diag(x)) + 1e-2
        # Extract the strictly upper triangular part of x
        xt = torch.tril(x)
        # Recombine into a new matrix with adjusted diagonal and original upper triangular parts
        return torch.diag(xd) + xt
    
    @property
    def R(self):
        # self.R_half.data = torch.tril(self.R_half.data)
        # jitter = torch.eye(self.v_dim).to(self.R_half.device) * 1e-6
        # return self.R_half @ self.R_half.T + jitter

        R_half = torch.tril(self.R_half)
        jitter = torch.eye(self.v_dim).to(self.R_half.device) * 1e-6
        return R_half @ R_half.T + jitter
    
        # R_half = self.reg(self.R_half)
        # return R_half @ R_half.T
    
    @property
    def Q(self):
        # self.B.data = torch.tril(self.B.data)
        # jitter = torch.eye(self.z_dim).to(self.B.device) * 1e-6
        # return self.B @ self.B.T + jitter

        B = torch.tril(self.B)
        jitter = torch.eye(self.z_dim).to(self.B.device) * 1e-6
        return B @ B.T + jitter

        # B = self.reg(self.B)
        # return B @ B.T
    
    @property
    def Sigma0(self):
        # self.Sigma0_half.data = torch.tril(self.Sigma0_half.data)
        # jitter = torch.eye(self.z_dim).to(self.Sigma0_half.device) * 1e-6
        # return self.Sigma0_half @ self.Sigma0_half.T + jitter

        Sigma0_half = torch.tril(self.Sigma0_half)
        jitter = torch.eye(self.z_dim).to(self.Sigma0_half.device) * 1e-6
        return Sigma0_half @ Sigma0_half.T + jitter

        # Sigma0_half = self.reg(self.Sigma0_half)
        # return Sigma0_half @ Sigma0_half.T
    

    def training_params(self, **kwargs): # code from mgp

        params = {
            'max_steps': 1001,
            'step_size': 100,
            'gamma': 0.85,
            'optimizer': optim.Adam,
            'batch_size': None,
            'print_every': 1,
            'lrate': 5E-2,
            'accumulate_gradient': True,
            'batch_mc': None,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        }

        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print('adding', key)

        if params['batch_size'] is None:
            params['batch_size'] = self.T

        return params
    
    def kalman_covariance(self, T=None): # return all matrices independent of observations
        if T is None:
            T = self.T
        _, Sigmas_diffused, Ks = general_kalman_covariance(self.A, self.W, self.Q, self.R, self.z_dim, self.v_dim, self.Sigma0, T, smoothing=False)
        return Sigmas_diffused, Ks
    
    def kalman_means(self, v: Tensor, Ks: Tensor):
        _, mus_diffused = general_kalman_means(self.A, self.W, self.z_dim, self.mu0, v[None, ...], Ks, Cs=None, smoothing=False)
        return mus_diffused.squeeze(1) # (T, ntrials, z_dim)

    def train_preprocessor(self, train_params):
        # So that the last dimension is the batch dimension
        transposed_v = self.v.transpose(0, -1) # (T, v_dim, ntrials)
        dataloader = DataLoader(transposed_v, batch_size=train_params['batch_size'])

        self.fit(dataloader, train_params)
    
    def fit(self, data: DataLoader, train_params):
        lrate = train_params['lrate']
        max_steps = train_params['max_steps']

        optimizer = train_params['optimizer']
        # optimizer = optimizer(self.parameters(), lr=lrate)
        optimizer = optimizer(self.parameters(), lr=lrate, betas=train_params['betas'], eps=train_params['eps'])
        scheduler = StepLR(optimizer, step_size=train_params['step_size'], gamma=train_params['gamma'])

        self.LLs = []
        for i in range(max_steps):
            for v_transposed in data:
                optimizer.zero_grad()
                v = v_transposed.transpose(-1, 0) # (ntrials, v_dim, batch_size)
                Sigmas_diffused, Ks = self.kalman_covariance() # (T-1, z_dim, z_dim), (T, z_dim, v_dim)
                mus_diffused = self.kalman_means(v, Ks) # (T, ntrials, z_dim)
                loss = -self.log_marginal(v, mus_diffused, Sigmas_diffused).sum()
                loss.backward()
                optimizer.step()
                self.LLs.append(-loss.item()/(self.v_dim * self.T * self.v.shape[0]))
            scheduler.step()
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])

    # returns p(v_{1:T})
    def log_marginal(self, v: Tensor, mus_diffused, Sigmas_diffused):
        # v is (ntrials, v_dim, T)
        # return log marginal likelihood of v
        # mus_diffused is (T-1, ntrials, z_dim)
        # Sigmas_diffused is (T-1, 1, z_dim, z_dim)
        # print(mus_diffused.shape, Sigmas_diffused.shape)
        Sigmas_diffused = Sigmas_diffused.squeeze(1) # (T-1, z_dim, z_dim)
        ntrials = v.shape[0]
        T = v.shape[-1]

        # ln p(v_1)
        m = self.W @ self.mu0
        cov = self.W @ self.Sigma0 @ self.W.T + self.R
        dist = MultivariateNormal(m, cov)
        first = dist.log_prob(v[..., 0]) # (ntrials, )

        # ln p(v_t|v_{1: t-1})
        ts = torch.arange(1, T).to(device)
        ms = (self.W @ mus_diffused[ts - 1][..., None]).squeeze(-1) # (T-1, ntrials, v_dim)
        # print(ms.shape)
        jitter = torch.eye(self.v_dim).to(self.R.device) * 1e-6
        covs = (self.W @ Sigmas_diffused[ts - 1] @ self.W.T) + self.R #+ jitter # (T-1, v_dim, v_dim)
        # print(covs.shape)
        covs_chol = torch.linalg.cholesky(covs)
        dists = MultivariateNormal(ms, scale_tril=covs_chol[:, None, ...])
        second = dists.log_prob(v.permute(2, 0, 1)[1:, ...]) # (T-1, ntrials)

        return first + second.sum(dim=0) # (ntrials, )
    
    def plot_LL(self):
        plt.plot(self.LLs)
        plt.xlabel('Step')
        plt.ylabel('LL')
        plt.show()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def sample_v(self, trials, T):
        z_0 = self.mu0[None, ...] + (torch.linalg.cholesky(self.Sigma0) @ torch.randn(trials, self.z_dim, 1).to(device)).squeeze(-1)
        z_s = torch.zeros(trials, self.z_dim, T, dtype=self.mu0.dtype).to(device)
        z_s[..., 0] = z_0
        samples = torch.zeros(trials, self.v_dim, T).to(device)
        for t in range(1, T):
            z_s[..., t] = (self.A @ z_s[..., t-1][..., None] + torch.linalg.cholesky(self.Q) @ torch.randn(trials, self.z_dim, 1).to(device)).squeeze(-1)
            samples[..., t] = (self.W @ z_s[..., t][..., None] + torch.linalg.cholesky(self.R) @ torch.randn(trials, self.v_dim, 1).to(device)).squeeze(-1)
        return samples
    
    def log_lik(self, z: Tensor, v: Tensor):
        # z is (n_mc, ntrials, z_dim, T)
        # return log likelihood of v given z averaged over MC samples

        mu = self.W @ z # (n_mc, ntrials, v_dim, T)
        mu = mu.transpose(-1, -2) # (n_mc, ntrials, T, v_dim)
        
        #TODO assume R_half is nice cholesky form
        dist = MultivariateNormal(mu, scale_tril=self.R_half[None, None, ...])
        v = v.transpose(-1, -2) # (ntrials, T, v_dim)
        ll = dist.log_prob(v[None, ...]) # (n_mc, ntrials, T)
        return ll.sum(axis=-1).mean(axis=0) # (ntrials,) 
