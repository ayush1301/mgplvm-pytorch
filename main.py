import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import abc
from torch.distributions import MultivariateNormal, Poisson
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[2]  # Return length along the last dimension

    def __getitem__(self, idx):
        return self.inputs[:, :, idx], self.outputs[:, :, idx]  # Return a tuple of (input, output)

    def my_collate_fn(batch):
        inputs, outputs = zip(*batch)  # Separate the inputs and outputs
        return torch.stack(inputs, dim=2), torch.stack(outputs, dim=2)  # Stack along a new last dimension

class GenerativeModel(Module, metaclass=abc.ABCMeta):
    def __init__(self, 
                 v: Tensor, 
                 Y: Tensor,
                 lik) -> None:
        '''
        v.shape = (ntrials, b, T)
        Y.shape = (ntrials, N, T)
        '''
        super().__init__()
        assert v.shape[0] == Y.shape[0]
        assert v.shape[2] == Y.shape[2]

        self.v = v.to(device)
        self.Y = Y.to(device)
        self.N = Y.shape[1]
        self.T = Y.shape[2]
        self.ntrials = Y.shape[0]
        self.b = v.shape[1]

        self.lik = lik

    def training_params(self, **kwargs): # code from mgp

        params = {
            'max_steps': 1001,
            'burnin': 150,
            'optimizer': optim.Adam,
            'batch_size': None,
            'print_every': 1,
            'lrate': 5E-2,
            'n_mc': 32,
            'accumulate_gradient': True,
            'batch_mc': None
        }

        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print('adding', key)

        if params['batch_size'] is None:
            params['batch_size'] = self.T

        return params

    def train_supervised_model(self, train_params):
        # dataset = TensorDataset(self.v, self.Y)
        # print(dataset[0][0].shape, dataset[0][1].shape)
        # print(dataset.tensors[0].shape, dataset.tensors[1].shape)
        # data = DataLoader(dataset, batch_size=train_params['batch_size'])
        # for v, y in data:
        #     print(v.shape, y.shape)

        dataset = MyDataset(self.v, self.Y)
        dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], collate_fn=MyDataset.my_collate_fn)

        self.fit(dataloader, train_params)

    def fit(self, data: DataLoader, train_params):
        lrate = train_params['lrate']
        n_mc = train_params['n_mc']
        max_steps = train_params['max_steps']

        optimizer = train_params['optimizer']
        optimizer = optimizer(self.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.85) # TODO: understand these parameters

        LLs = []
        for i in range(max_steps):
            prev_v = None
            for v, y in data: # loop over batches
                loss = -self.joint_LL(n_mc, v, y, prev_v).mean() # TODO: should I use mean??
                loss.backward()
                LLs.append(-loss.item())
                optimizer.step()
                optimizer.zero_grad()
                prev_v = v[..., -1] # (ntrials, b)
            scheduler.step()
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', LLs[-1])


class LDS(GenerativeModel):
    def __init__(self, v: Tensor, Y: Tensor, lik) -> None:
        super().__init__(v, Y, lik)

        # Generative parameters
        A = torch.randn(1, self.b, self.b).to(device)
        C = torch.randn(1, self.N, self.b).to(device)
        B = torch.randn(1, self.b, self.b).to(device)
        mu0 = torch.randn(1, self.b).to(device)
        Sigma0_half = torch.eye(self.b)[None, ...].to(device)
        sigma_x = torch.randn(1).to(device)

        print(A, B)

        self.A = torch.nn.Parameter(A)
        self.C = torch.nn.Parameter(C)
        self.B = torch.nn.Parameter(B)
        self.sigma_x = torch.nn.Parameter(sigma_x)
        self.mu0 = torch.nn.Parameter(mu0)
        self.Sigma0_half = torch.nn.Parameter(Sigma0_half)

        # print(self.A.shape, self.B.shape, self.C.shape, self.sigma_x.shape, self.mu0.shape, self.Sigma0.shape)
    
    @property
    def var_x(self):
        return torch.square(self.sigma_x)

    @property
    def Q(self):
        Q = self.B @ self.B.transpose(-1, -2)
        jitter = torch.eye(Q.shape[-1]).to(Q.device) * 1e-6 
        return Q + jitter

    @property
    def Sigma0(self):
        Sigma0 = self.Sigma0_half @ self.Sigma0_half.transpose(-1, -2)
        jitter = torch.eye(Sigma0.shape[-1]).to(Sigma0.device) * 1e-6
        return Sigma0 + jitter
    
    def joint_LL(self, n_mc, v, Y, 
                 prev_v # (ntrials, b)
                 ):
        # Natural parameters for p(x_t|v_t)
        ntrials, N, T = Y.shape # Only T is different from self.Y.shape
        _, b, _ = v.shape

        mu = self.C @ v # (ntrials, N, T)

        # R_half = torch.eye(self.N)[None, ...] * self.sigma_x[:, None, None] # (ntrials, N, N)
        # # print(mu.shape, R_half.shape)
        # samples = torch.randn(n_mc, self.ntrials, self.N, self.T)
        # # After expansion, the shape is (n_mc, ntrials, N, N)
        # samples = R_half[None, ...].expand(samples.shape[0], -1, -1, -1) @ samples + mu[None, ...]
        # # print(samples.shape)

        samples = torch.randn(n_mc, ntrials, N, T).to(device)
        # sigma_expanded = self.sigma_x[None,:]#.expand(samples.shape[0], -1) # (n_mc, ntrials)
        # samples = sigma_expanded[... , None, None] * samples + mu[None, ...]
        samples = self.sigma_x * samples + mu[None, ...]
        # print(samples.shape)

        first = self.lik.LL(samples, Y) # (ntrials,)
        # print(first.shape)

        ## Second term of LL
        v0 = v[..., 0] # (ntrials, b)
        if prev_v is None:
            # print(self.mu0.shape, self.Sigma0.shape)
            second_small = MultivariateNormal(self.mu0, self.Sigma0).log_prob(v0) # (ntrials, )
        else:
            # p(v_0|v_{-1})
            mu = self.A @ prev_v[...,None] # (ntrials, b, 1)
            mu = mu[..., 0] # (ntrials, b)
            # print(mu.shape, self.Q.shape)
            second_small = MultivariateNormal(mu, self.Q).log_prob(v0) # (ntrials, )
        # print(second_small.shape)

        # Natural parameters for p(v_t|v_{t-1})
        mus = self.A @ v[... , :-1] # (ntrials, b, T-1)
        # print(mu.shape, self.Q.shape)

        # second_big = torch.zeros(ntrials).to(device)
        # for t in range(T-1): # TODO, can we vectorize this?
        #     dist = MultivariateNormal(mus[..., t], self.Q)
        #     second_big += dist.log_prob(v[..., t+1])

        # Replace the for loop with this code

        dist = MultivariateNormal(mus.transpose(-1, -2), self.Q[:, None, :, :])
        second_big = dist.log_prob(v[..., 1:].transpose(-1,-2)).sum(dim=-1)


        # print(second_big.shape)
        second = second_small + second_big

        return first + second



class Poisson_noise():
    def __init__(self, link_fn=torch.exp, d=0, fixed_d=True) -> None:
        self.link_fn = link_fn
        self.d = torch.nn.Parameter(torch.tensor(d), requires_grad=not fixed_d)

    def LL(self, x, y) -> Tensor:
        '''
        x.shape = (n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        rates = self.link_fn(x)
        dist = Poisson(rates)
        log_prob = dist.log_prob(y[None, ...]) # (n_mc, ntrials, N, T)

        # avg_log_prob = torch.mean(log_prob, dim=0)  # (ntrials, N, T)
        # total_log_prob = torch.sum(avg_log_prob, dim=(1,2)) # (ntrials, )

        avg_log_prob = torch.logsumexp(log_prob, dim=0) - np.log(log_prob.shape[0]) # (ntrials, N, T)
        total_log_prob = torch.sum(avg_log_prob, dim=(1,2)) # (ntrials, )

        return total_log_prob

class Gaussian_noise():
    def __init__(self, sigma) -> None:
        self.sigma = sigma



