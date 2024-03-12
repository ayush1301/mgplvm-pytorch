import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import abc
from torch.distributions import MultivariateNormal, Poisson, NegativeBinomial, Normal
from itertools import chain
import torch.distributions as dists
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import Module
from torch.optim.lr_scheduler import StepLR, LambdaLR
from preprocessor import Preprocessor
from utils import general_kalman_covariance, general_kalman_means

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
                 z: Tensor, 
                 Y: Tensor,
                 lik) -> None:
        '''
        z.shape = (ntrials, b, T)
        Y.shape = (ntrials, N, T)
        '''
        super().__init__()
        assert z.shape[0] == Y.shape[0]
        assert z.shape[2] == Y.shape[2]

        self.z = z.to(device)
        self.Y = Y.to(device)
        self.N = Y.shape[1]
        self.T = Y.shape[2]
        self.ntrials = Y.shape[0]
        self.b = z.shape[1]

        self.lik = lik

    def training_params(self, **kwargs): # code from mgp

        params = {
            'max_steps': 1001,
            'step_size': 100,
            'gamma': 0.85,
            'optimizer': optim.Adam,
            'batch_size': None,
            'print_every': 1,
            'lrate': 5E-2,
            'n_mc': 32,
            'accumulate_gradient': True,
            'batch_mc': None,
            'burnin': 100,
            'StepLR': True,
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
        dataset = MyDataset(self.z, self.Y)
        dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], collate_fn=MyDataset.my_collate_fn, shuffle=False)

        self.fit(dataloader, train_params)

    def fit(self, data: DataLoader, train_params):
        # set learning rate schedule so sigma updates have a burn-in period
        burnin = train_params['burnin']
        def fburn(x):
            return 1 - np.exp(-x / (3 * burnin))
        lrate = train_params['lrate']
        n_mc = train_params['n_mc']
        max_steps = train_params['max_steps']

        optimizer = train_params['optimizer']
        optimizer = optimizer(self.prms, lr=lrate)
        if train_params['StepLR']:
            scheduler = StepLR(optimizer, step_size=train_params['step_size'], gamma=train_params['gamma']) # TODO: understand these parameters
        else:
            scheduler = LambdaLR(optimizer, lr_lambda=[fburn])

        self.LLs = []
        for i in range(max_steps):
            prev_z = None
            loss_vals = []
            for z, y in data: # loop over batches
                loss = -self.joint_LL(n_mc, z, y, prev_z).sum() # TODO: should I use mean??
                loss.backward()
                loss_vals.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                prev_z = z[..., -1] # (ntrials, b)
            scheduler.step()
            Z = self.T * self.ntrials * self.N + self.T * self.ntrials * self.b # TODO: check this
            self.LLs.append(-np.sum(loss_vals)/Z)
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])
    
    @property
    def prms(self):
        # return chain(self.parameters(), self.lik.parameters())
        return self.parameters()
    
    def freeze_params(self):
        for param in self.prms:
            param.requires_grad = False

    def plot_LL(self):
        plt.plot(self.LLs)
        plt.xlabel('Step')
        plt.ylabel('LL')
        plt.show()


class LDS(GenerativeModel):
    def __init__(self, z: Tensor, Y: Tensor, lik, x_dim=None, link_fn=torch.exp,
                 A=None, C=None, W=None, B=None, mu0=None, Sigma0_half=None, sigma_x=None,
                 trained_z=False, d=0., fixed_d=True, single_sigma_x=False, full_R=False, analytical_init=False) -> None:
        super().__init__(z, Y, lik)

        self.link_fn = link_fn

        if x_dim is None:
            self.x_dim = self.b
        else:
            self.x_dim = x_dim

        # Generative parameters
        if A is None:
            A = 0.8 * torch.randn(1, self.b, self.b).to(device) / np.sqrt(self.b)
            # A = torch.eye(self.b)[None, ...].to(device)

        if C is None:
            C = torch.randn(1, self.N, self.x_dim).to(device) / np.sqrt(self.x_dim)

            # # Create an identity matrix of size x_dim
            # identity = torch.eye(self.x_dim).to(device)

            # # Repeat the identity matrix N//x_dim times
            # C = identity.repeat(self.N//self.x_dim, 1)

            # # If N is not a multiple of x_dim, append the remaining rows with zeros
            # if self.N % self.x_dim != 0:
            #     zeros = torch.zeros((self.N % self.x_dim, self.x_dim)).to(device)
            #     C = torch.cat((C, zeros))

            # # Reshape C to have shape (1, N, x_dim)
            # C = C.unsqueeze(0)

        if W is None:
            W = torch.randn(1, self.x_dim, self.b).to(device) / np.sqrt(self.b)

            # # Create an identity matrix of size x_dim
            # identity = torch.eye(self.x_dim).to(device)

            # # Slice the identity matrix to get the first b columns
            # W = identity[:, :self.b]

            # # Reshape W to have shape (1, x_dim, b)
            # W = W.unsqueeze(0)
        if B is None:
            # B = torch.rand(1, self.b, self.b).to(device)
            B = torch.eye(self.b)[None, ...].to(device)
        if mu0 is None:
            # mu0 = torch.rand(1, self.b).to(device)
            mu0 = torch.zeros(self.b).to(device)
        if Sigma0_half is None:
            Sigma0_half = 0.1 * torch.eye(self.b)[None, ...].to(device)
        if sigma_x is None:
            sigma_x = 0.1 * torch.ones(self.x_dim).to(device)

        self.A = torch.nn.Parameter(A, requires_grad= not trained_z)
        self.C = torch.nn.Parameter(C)
        self.W = torch.nn.Parameter(W)
        self.B = torch.nn.Parameter(B, requires_grad= not trained_z)
        if full_R:
            R_half = 0.1 * torch.eye(self.x_dim).to(device)
            self.R_half = torch.nn.Parameter(R_half)
        else:
            self.log_sigma_x = torch.nn.Parameter(torch.log(sigma_x))
        self.mu0 = torch.nn.Parameter(mu0, requires_grad= not trained_z)
        self.Sigma0_half = torch.nn.Parameter(Sigma0_half, requires_grad= not trained_z)
        # self.d = torch.nn.Parameter(torch.tensor(d).to(device), requires_grad=not fixed_d)
        self.d = torch.nn.Parameter(d * torch.ones(self.N).to(device), requires_grad=not fixed_d)

        self.single_sigma_x = single_sigma_x
        self.full_R = full_R

        if analytical_init:
            self.analytical_init()
        # print(self.A.shape, self.B.shape, self.C.shape, self.sigma_x.shape, self.mu0.shape, self.Sigma0.shape)
    
    # @property
    # def B(self):
    #     return torch.tril(self._B)
    
    @property
    def sigma_x(self):
        if self.single_sigma_x:
            return torch.ones(self.x_dim).to(device) * torch.exp(self.log_sigma_x[0])
        else:
            return torch.exp(self.log_sigma_x)

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
    
    @property
    def R(self):
        # R =  self.var_x * torch.eye(self.x_dim).unsqueeze(0).to(device) # (1, x_dim, x_dim)
        if self.full_R:
            R_half = torch.tril(self.R_half, diagonal=-1) + torch.diag(torch.exp(torch.diag(self.R_half)))
            R = (R_half @ R_half.T).unsqueeze(0).to(device) # (1, x_dim, x_dim)
            # R = (self.R_half @ self.R_half.T).unsqueeze(0).to(device) # (1, x_dim, x_dim)
        else:
            R = torch.diag(self.var_x).unsqueeze(0).to(device) # (1, x_dim, x_dim) 
        jitter = torch.eye(R.shape[-1]).to(R.device) * 1e-6
        return R + jitter
    
    def joint_LL(self, 
                 n_mc, # number of samples for x
                 z, # (ntrials, b, T) OR (n_mc_z, ntrials, b, T)
                 Y, # (ntrials, N, T)
                 prev_z, # (ntrials, b) OR (n_mc_z, ntrials, b)
                 ret_only_joint=True
                 ):
        # Natural parameters for p(x_t|v_t)
        ntrials, N, T = Y.shape # Only T is different from self.Y.shape

        # Adjust tensor shapes for n_mc_z
        W = self.W[None, ...]
        C = self.C[None, ...]
        A = self.A[None, ...]
        Q = self.Q[None, ...]
        if len(z.shape) == 3: # When training the gen model so modifying z such that n_mc_z = 1
            z = z[None, ...] # (1, ntrials, b, T)
        n_mc_z = z.shape[0]

        # Calculate first term of LL (p(y_{1:T}|z_{1:T}))
        mu = W @ z # (n_mc_z, ntrials, x_dim, T)
        samples = torch.randn(n_mc, n_mc_z, ntrials, self.x_dim, T).to(device)

        # samples = self.sigma_x * samples + mu[None, ...]
        # samples = torch.diag(self.sigma_x) @ samples + mu[None, ...] # (n_mc, n_mc_z, ntrials, x_dim, T)
        samples = torch.linalg.cholesky(self.R.squeeze(0)) @ samples + mu[None, ...] # (n_mc, n_mc_z, ntrials, x_dim, T)

        # print(samples.shape)
        firing_rates = self.link_fn(C[None, ...] @ samples + self.d[:, None]) # (n_mc_z, n_mc, ntrials, N, T)
        first = self.lik.LL(firing_rates, Y) # (ntrials,)
        # print(first.shape)

        ## Second term of LL p(z_{1:T})
        z0 = z[..., 0] # (n_mc_z, ntrials, b)
        if prev_z is None:
            # print(self.mu0.shape, self.Sigma0.shape)
            second_small = MultivariateNormal(self.mu0, self.Sigma0).log_prob(z0).mean(dim=0) # (ntrials, ) (Averaging over n_mc_z)
        else:
            # p(z_0|z_{-1})
            if len(prev_z.shape) == 2: # When training the gen model so modifying prev_z such that n_mc_z = 1
                prev_z = prev_z[None, ...] # (1, ntrials, b)

            mu = A @ prev_z[...,None] # (n_mc_z, ntrials, b, 1)
            mu = mu[..., 0] # (m_mc_z, ntrials, b)
            # print(mu.shape, self.Q.shape)
            second_small = MultivariateNormal(mu, Q).log_prob(z0).mean(dim=0) # (ntrials, ) (Averaging over n_mc_z) # TODO: consider usong scale tril for speed
        # print(second_small.shape)

        # Natural parameters for p(z_t|z_{t-1})
        mus = A @ z[... , :-1] # (n_mc_z, ntrials, b, T-1)
        # print(mu.shape, self.Q.shape)

        dist = MultivariateNormal(mus.transpose(-1, -2), Q[:, :, None, :, :])
        second_big = dist.log_prob(z[..., 1:].transpose(-1,-2)).sum(dim=-1).mean(dim=0) # (ntrials, ) (Averaging over n_mc_z)


        # print(second_big.shape)
        second = second_small + second_big
        if ret_only_joint:
            return first + second
        else:
            return first+second, first, second # p(y_{1:T}|z_{1:T}), p(z_{1:T})
    
    def sample_z(self, n_mc: int, trials, prev_z=None): # Sample z from the prior of the LDS
        # prev_z is (n_mc, ntrials, b)
        samples = torch.zeros(n_mc, trials, self.b, self.T).to(device)
        start_t = 0
        if prev_z is None:
            # TODO: optimise cholesky?
            z0 = self.mu0[None, ...] + (torch.linalg.cholesky(self.Sigma0)[None, ...] @ torch.randn(n_mc, trials, self.b, 1).to(device)).squeeze(-1) # (n_mc, ntrials, b)
            samples[..., 0] = z0
            start_t = 1
            prev_z = z0
        for t in range(start_t, self.T):
            z_t = (self.A[None, ...] @ prev_z[..., None] + torch.linalg.cholesky(self.Q)[None, ...] @ torch.randn(n_mc, trials, self.b, 1).to(device)).squeeze(-1)
            samples[..., t] = z_t
            prev_z = z_t

        return samples

    # Initialise W and R assuming Gaussian noise model
    def analytical_init(self):
        assert self.x_dim == self.N # x_dim = N for analytical initialisation
        y = self.Y - self.Y.mean(dim=(0,-1), keepdim=True)
        y = y.transpose(-1, -2) # (ntrials, T, N)
        
        z = self.z.transpose(-1, -2) # (ntrials, T, b)
        
        outer_yz = (y[..., None] @ z[..., None, :]).sum(dim=(0, 1)) # (N, b)
        outer_zz = (z[..., None] @ z[..., None, :]).sum(dim=(0, 1)) # (b, b)
        W = outer_yz @ torch.linalg.inv(outer_zz) # (N, b)

        outer_yy = (y[..., None] @ y[..., None, :]).sum(dim=(0, 1)) # (N, N)
        outer_zy = (z[..., None] @ y[..., None, :]).sum(dim=(0, 1)) # (b, N)
        R = (outer_yy - W @ outer_zy)/(self.T * self.ntrials) # (N, N)

        self.W.data = W[None, ...] # (1, N, b)
        R_half = torch.linalg.cholesky(R)
        log_diag_R_half = torch.log(torch.diag(R_half))
        
        if self.full_R:
            self.R_half.data = torch.tril(R_half, diagonal=-1) + torch.diag(log_diag_R_half)
        else:
            assert self.single_sigma_x == False
            self.log_sigma_x.data = log_diag_R_half



class Noise(Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def general_LL(self, dist, y):
        # y.shape = (ntrials, N, T)
        log_prob = dist.log_prob(y[None, None, ...]) # (n_mc_z, n_mc, ntrials, N, T)

        avg_log_prob = torch.logsumexp(log_prob, dim=(0,1)) - np.log(log_prob.shape[0] * log_prob.shape[1]) # (ntrials, N, T)
        total_log_prob = torch.sum(avg_log_prob, dim=(-1, -2))

        return total_log_prob


class Negative_binomial_noise(Noise):
    def __init__(self, Y: Tensor) -> None:
        super().__init__()
        # ntrials, N, T = Y.shape
        total_count = torch.tensor(torch.mean(Y, dim=(0, 2))).to(device) # (N, )
        total_count = dists.transform_to(
            dists.constraints.greater_than_eq(0)).inv(total_count)
        self._total_count = torch.nn.Parameter(total_count)
    
    @property
    def total_count(self):
        return dists.transform_to(dists.constraints.greater_than_eq(0))(
            self._total_count)
    
    def LL(self, rates, y) -> Tensor:
        '''
        rates.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = NegativeBinomial(total_count=self.total_count[None, None, None, :, None], logits=rates)
        return self.general_LL(dist, y)
        
class Poisson_noise(Noise):
    def __init__(self) -> None:
        super().__init__()
        
    def LL(self, rates, y) -> Tensor:
        '''
        rates.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = Poisson(rates + 1e-6) # TODO: is this a good idea? (adding small number to avoid log(0))
        return self.general_LL(dist, y)

class Gaussian_noise(Noise):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.log_sigma = torch.nn.Parameter(torch.tensor(np.log(sigma)).to(device))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def LL(self, rates, y) -> Tensor:
        '''
        rates.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = Normal(rates, self.sigma)
        return self.general_LL(dist, y)

class MultiHeadNetwork(Module):
    def __init__(self, input_size, shared_layer_sizes, head_layer_sizes, head_names):
        super(MultiHeadNetwork, self).__init__()

        # Define shared layers
        self.shared_layers = self._make_layers(input_size, shared_layer_sizes, final_relu=True)
        self.head_names = head_names

        # Define separate heads
        self.heads = torch.nn.ModuleList([
            self._make_layers(shared_layer_sizes[-1], layers, final_relu=False)
            for layers in head_layer_sizes
        ])

    def _make_layers(self, input_size, layer_sizes, final_relu):
        layers = []
        for i, layer_size in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(input_size, layer_size))
            if i != len(layer_sizes) - 1 or final_relu:
                layers.append(torch.nn.ReLU())
            input_size = layer_size
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_layers(x)
        ret= {}
        for i, head in enumerate(self.heads):
            ret[self.head_names[i]] = head(x)
    
class RNNModel(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

class RecognitionModel(Module):
    def __init__(self, gen_model: LDS, hidden_layer_size: int = 100, neural_net=None, rnn=False, cov_change=False, zero_mean_x_tilde=False, preprocessor: Preprocessor = None, gen_model_fixed=True) -> None:
        super(RecognitionModel, self).__init__()
        self.gen_model = gen_model
        # Define a 2 layer MLP with hidden_layer_size hidden units
        if neural_net is None:
            if rnn:
                self.neural_net = RNNModel(gen_model.N, hidden_layer_size, gen_model.x_dim).to(device)
            elif cov_change:
                self.neural_net = MultiHeadNetwork(
                    input_size=gen_model.N,
                    shared_layer_sizes=[100,],
                    head_names=['x_tilde', 'R'],
                    head_layer_sizes=[[gen_model.x_dim,], [gen_model.x_dim,]]
                )
                # self.neural_net = MultiHeadRNN(
                #     input_size=gen_model.N,
                #     hidden_size=hidden_layer_size,
                #     output_sizes={'x_tilde': gen_model.x_dim, 'delta_R': gen_model.x_dim, 'delta_W': gen_model.x_dim * gen_model.b}
                # ).to(device)
            else:
                self.neural_net = torch.nn.Sequential(
                    torch.nn.Linear(gen_model.N, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, gen_model.x_dim)
                ).to(device)
        else:
            self.neural_net = neural_net.to(device)
        self.cov_change = cov_change # Time varying LDS
        self.zero_mean_x_tilde = zero_mean_x_tilde

        self.preprocessor = preprocessor
        self.gen_model_fixed = gen_model_fixed # False if the generative model is also being trained

        # For debugging
        self.dWs = []
        self.dRs = []
    def training_params(self, **kwargs): # code from mgp
        params = {
            'max_steps': 1001,
            'step_size': 100,
            'gamma': 0.85,
            'optimizer': optim.Adam,
            'batch_size': None,
            'print_every': 1,
            'lrate': 5E-2,
            'n_mc_x': 32,
            'n_mc_z': 32,
            'accumulate_gradient': True,
            'batch_mc_z': None,
            'print_deltas': False,
        }

        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print('adding', key)

        if params['batch_size'] is None:
            params['batch_size'] = self.gen_model.ntrials

        return params
    
    def gen_model_R(self, pseudo_obs):
        if not self.cov_change or pseudo_obs is None or pseudo_obs['delta_R'] is None:
            return self.gen_model.R.squeeze(0)
        else:
            if not self.gen_model.full_R:
                # delta_R directly changes the diagonal elements of R

                # # Get the diagonal and off-diagonal elements
                # delta_R = pseudo_obs['delta_R'] # (batch_size, ntrials, x_dim)
                # # diag = torch.diag(self.gen_model.R.squeeze(0) + torch.diag(delta_R))
                # diag = torch.diag(self.gen_model.R.squeeze(0)) + delta_R # (batch_size, ntrials, x_dim)
                # # off_diag = self.gen_model.R.squeeze(0) + torch.diag(delta_R) - torch.diag(diag)
                
                # # Clamp the diagonal elements
                # diag = diag.clamp_min(1e-6)

                # return torch.diag_embed(diag) # (batch_size, ntrials, x_dim, x_dim)
                # # Add the clamped diagonal elements back to the off-diagonal elements
                # # R = off_diag + torch.diag(diag)
                
                # delta_R changes log(sigma_x) #TODO: not compatible with single_sigma_x
                delta_R = pseudo_obs['delta_R'] # (batch_size, ntrials, x_dim)
                # print(torch.max(delta_R), torch.min(delta_R))
                new_log_sigma_x = self.gen_model.log_sigma_x + delta_R
                new_var_x = torch.square(torch.exp(new_log_sigma_x)) + 1e-6
                self.dRs.append((torch.max(delta_R).item(), torch.min(delta_R).item()))
                return torch.diag_embed(new_var_x) # (batch_size, ntrials, x_dim, x_dim)

            else:
                # Assume implementation of full_R where diag elements are exponentiated
                delta_R = pseudo_obs['delta_R'] # (batch_size, ntrials, x_dim)
                if delta_R.shape[-1] == self.gen_model.x_dim:
                    delta_R = torch.diag_embed(torch.exp(delta_R)) # (batch_size, ntrials, x_dim, x_dim)
                elif delta_R.shape[-1] == self.gen_model.x_dim * (self.gen_model.x_dim + 1) / 2:
                    delta_R_reshaped = torch.zeros(delta_R.shape[0], delta_R.shape[1], self.gen_model.x_dim, self.gen_model.x_dim).to(device)
                    rows, cols = torch.tril_indices(self.gen_model.x_dim, self.gen_model.x_dim)
                    delta_R_reshaped[..., rows, cols] = delta_R
                    delta_R = torch.tril(delta_R_reshaped, diagonal=-1) + torch.diag_embed(torch.exp(torch.diagonal(delta_R_reshaped, dim1=-2, dim2=-1))) # (batch_size, ntrials, x_dim, x_dim)
                else:
                    raise ValueError('delta_R has wrong shape')
                    
                self.dRs.append((torch.max(delta_R).item(), torch.min(delta_R).item()))
                R_half = torch.tril(self.gen_model.R_half, diagonal=-1) + torch.diag_embed(torch.exp(torch.diag(self.gen_model.R_half))) + delta_R # (batch_size, ntrials, x_dim, x_dim)
                # new_log_diag = torch.diag(self.gen_model.R_half) + delta_R # (batch_size, ntrials, x_dim)
                # new_diag = torch.exp(new_log_diag)
                # R_half = torch.tril(self.gen_model.R_half, diagonal=-1) + torch.diag_embed(new_diag)
                return R_half @ R_half.transpose(-1,-2) + 1e-6 * torch.eye(self.gen_model.x_dim).to(device) # (batch_size, ntrials, x_dim, x_dim)
            
    def gen_model_W(self, pseudo_obs):
        if not self.cov_change or pseudo_obs is None or pseudo_obs['delta_W'] is None:
            return self.gen_model.W.squeeze(0)
        else:
            # delta_W directly changes the W matrix
            delta_W = pseudo_obs['delta_W'] # (batch_size, ntrials, x_dim, b)
            self.dWs.append((torch.max(delta_W).item(), torch.min(delta_W).item()))
            # print(torch.max(delta_W), torch.min(delta_W))
            
            return self.gen_model.W.squeeze(0) + delta_W

    def train_full_model(self, train_params_gen, train_params_recognition):
        self.gen_model.train_supervised_model(train_params_gen)
        self.gen_model.freeze_params()
        self.train_recognition_model(train_params_recognition)
        
    def train_recognition_model(self, train_params_recognition):
        if self.preprocessor is None:
            dataloader = DataLoader(self.gen_model.Y, batch_size=train_params_recognition['batch_size'])
        else:
            dataset = TensorDataset(self.gen_model.Y, self.preprocessor.v)
            dataloader = DataLoader(dataset, batch_size=train_params_recognition['batch_size'])

        self.fit(dataloader, train_params_recognition)
    
    def get_x_tilde(self, y: Tensor, only_x_tilde=True):
        delta_R = None
        delta_W = None
        # y is (ntrials, N, batch_size)
        if not self.cov_change:
            x_tilde = self.neural_net(y.transpose(-1, -2)) # (ntrials, batch_size, x_dim)
        else:
            # tilde = self.neural_net(y.transpose(-1, -2)) # (ntrials, batch_size, x_dim+1)
            # # Assume for now that the first x_dim elements are for x_tilde and the next element is for R
            # x_tilde = tilde[..., :self.gen_model.x_dim] # (ntrials, batch_size, x_dim)

            # delta_R = tilde[..., self.gen_model.x_dim:] # (ntrials, batch_size, x_dim)
            # delta_R = delta_R.transpose(1,0) # (batch_size, ntrials, x_dim)

            # Assume neural_net outputs dictionary
            pseudo_obs = self.neural_net(y.transpose(-1, -2)) # {'x_tilde': (ntrials, batch_size, x_dim), 'delta_R': (ntrials, batch_size, x_dim), 'delta_W': (ntrials, batch_size, x_dim * b)}
            x_tilde = pseudo_obs['x_tilde']
            if 'delta_R' in pseudo_obs:
                delta_R = pseudo_obs['delta_R'].transpose(1,0) # (batch_size, ntrials, x_dim)
            if 'delta_W' in pseudo_obs:
                delta_W = pseudo_obs['delta_W'].transpose(1,0) # (batch_size, ntrials, x_dim * b)
                delta_W = delta_W.reshape(delta_W.shape[0], delta_W.shape[1], self.gen_model.x_dim, self.gen_model.b) # (batch_size, ntrials, x_dim, b)


        x_tilde =  x_tilde.transpose(-1, -2) # (ntrials, x_dim, batch_size)
        if self.zero_mean_x_tilde:
            x_tilde = x_tilde - x_tilde.mean(dim=(0,2), keepdim=True)
        if only_x_tilde:
            return x_tilde
        else:
            return {'x_tilde': x_tilde, 'delta_R': delta_R, 'delta_W': delta_W}
    
    def sample_matheron_pert(self, n_mc: int, trials, pseudo_obs=None):
        z_prior = self.gen_model.sample_z(n_mc, trials).transpose(-1, -2) # (n_mc, ntrials, T, b) # TODO: how to deal with batch_size?
        W = self.gen_model_W(pseudo_obs)
        if len(W.shape) != 2:
            W = W.transpose(0, 1) # (ntrials, T, x_dim, b)
        pertubation = W @ z_prior[..., None] # (n_mc, ntrials, T, x_dim, 1)
        # pertubation = self.gen_model.W[None, ...] @ z_prior[..., None] # (n_mc, ntrials, T, x_dim, 1)
        # noise = torch.linalg.cholesky(self.gen_model.R) @ torch.randn(*pertubation.shape).to(device) # (n_mc, ntrials, T, x_dim, 1)
        R = self.gen_model_R(pseudo_obs)
        if len(R.shape) != 2:
            R = R.transpose(0, 1) # (ntrials, T, x_dim, x_dim)
        noise = torch.linalg.cholesky(R) @ torch.randn(*pertubation.shape).to(device) # (n_mc, ntrials, T, x_dim, 1)
        return (pertubation + noise).squeeze(-1).transpose(-1, -2) # (n_mc, ntrials, x_dim, T)
    
    def kalman_covariance(self, T=None, get_sigma_tilde=False, pseudo_obs=None): # return all matrices independent of observations
        if T is None:
            T = self.gen_model.T
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model_W(pseudo_obs)
        Q = self.gen_model.Q.squeeze(0) # (b, b)
        R = self.gen_model_R(pseudo_obs)
        Sigma0 = self.gen_model.Sigma0.squeeze(0) # (b, b)
        b = self.gen_model.b
        x_dim = self.gen_model.x_dim
        return general_kalman_covariance(A, W, Q, R, b, x_dim, Sigma0, T, get_sigma_tilde)
    
    def kalman_means(self, x_hat: Tensor, Ks: Tensor, Cs: Tensor, pseudo_obs=None):
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model_W(pseudo_obs)
        b = self.gen_model.b
        mu0 = self.gen_model.mu0.squeeze(0) # (b)
        return general_kalman_means(A, W, b, mu0, x_hat, Ks, Cs)
    

    def entropy(self, samples, mus_filt, mus_diffused, Cs, Sigmas_tilde_chol):
        # samples is (batch_size, n_mc_z, ntrials, b)
        # mus_filt is (batch_size, n_mc_z, ntrials, b)
        # mus_diffused is (batch_size-1, n_mc_z, ntrials, b)
        # Cs is (batch_size-1, ntrials, b, b) (ntrials can be 1 for const covariance)
        # Sigmas_tilde_chol is (batch_size, ntrials, b, b) (ntrials can be 1 for const covariance)

        batch_size, n_mc_z, n_trials, b = samples.shape
        mus_tilde = torch.zeros(batch_size, n_mc_z, n_trials, b).to(device) # mu_tilde[t] = E[z_t|z_{t+1:T}, y_{1:T}]
        mus_tilde[-1] = mus_filt[-1] # (n_mc_z, ntrials, b)
        t_values = torch.arange(batch_size - 2, -1, -1)
        mus_tilde[t_values] = mus_filt[t_values] + (Cs[:, None, ...][t_values] @ (samples[t_values + 1] - mus_diffused[t_values])[..., None]).squeeze(-1)
        
        dist = MultivariateNormal(mus_tilde, scale_tril=Sigmas_tilde_chol[:, None, ...])
        # dist = MultivariateNormal(mus_tilde, covariance_matrix=Sigmas_tilde[:, None, None, ...])
        log_prob = dist.log_prob(samples).sum(dim=0) # (n_mc_z, ntrials)
        return -log_prob.mean(dim=0) # (ntrials,)


    def fit(self, data: DataLoader, train_params_recognition):
        lrate = train_params_recognition['lrate']
        n_mc_x = train_params_recognition['n_mc_x']
        n_mc_z = train_params_recognition['n_mc_z']
        batch_mc_z = train_params_recognition['batch_mc_z']
        max_steps = train_params_recognition['max_steps']
        accumulate_gradient = train_params_recognition['accumulate_gradient']

        optimizer = train_params_recognition['optimizer']
        # optimizer = optimizer(self.neural_net.parameters(), lr=lrate)
        optimizer = optimizer(self.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params_recognition['step_size'], gamma=train_params_recognition['gamma'])

        if batch_mc_z is None:
            batch_mc_z = n_mc_z
        mc_batches = [batch_mc_z for _ in range(n_mc_z // batch_mc_z)]
        if (n_mc_z % batch_mc_z) > 0:
            mc_batches.append(n_mc_z % batch_mc_z)
        assert np.sum(mc_batches) == n_mc_z

        self.LLs = []
        self.entropy_vals = []
        self.y_LL_vals = []
        self.prior_LL_vals = []
        self.v_LL_vals = []
        if self.gen_model_fixed and not self.cov_change:
            _ , _, Ks, Cs, Sigmas_tilde_chol = self.kalman_covariance(get_sigma_tilde=True) # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b), (T, b, b)

        for i in range(max_steps):
            loss_vals = []
            entropy_vals = []
            prior_LL_vals = []
            y_LL_vals = []
            v_LL_vals = []
            prev_mu = None
            prev_Sigma = None
            prev_z = None
            for batch_mc_z in mc_batches:
                mc_weight = batch_mc_z / n_mc_z # fraction of the total samples
                for _data in data: # loop over batches TODO
                    if isinstance(_data, list):
                        y = _data[0]
                        v = _data[1]
                    else:
                        y = _data
                        v = None
                    batch_trials = y.shape[0]
                    # batch_weight = batch_trials / self.gen_model.ntrials
                    batch_weight = 1
                    # NN pseudo observations
                    # y = y.transpose(-1, 0) # (ntrials, N, batch_size)
                    pseudo_obs = self.get_x_tilde(y, only_x_tilde=False)
                    x_tilde = pseudo_obs['x_tilde'] # (ntrials, x_dim, batch_size)
                    # Matheron psuedo observations
                    matheron_pert = self.sample_matheron_pert(batch_mc_z, batch_trials, pseudo_obs) # (n_mc_z, ntrials, x_dim, T) # TODO: do I need to do this every time?
                    x_hat = x_tilde[None, ...] - matheron_pert[..., :x_tilde.shape[-1]] # (n_mc_z, ntrials, x_dim, batch_size) TODO: how to deal with batch_size?
                    if self.cov_change or not self.gen_model_fixed:
                        _ , _, Ks, Cs, Sigmas_tilde_chol = self.kalman_covariance(get_sigma_tilde=True, pseudo_obs=pseudo_obs) # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b), (T, b, b)
                    loss, entropy, y_LL, prior_LL, v_LL = self.LL(n_mc_x, x_hat, y, Ks, Cs, Sigmas_tilde_chol, pseudo_obs=pseudo_obs, v=v)
                    loss *= -1 # negative LL

                    if accumulate_gradient:
                        loss = loss * mc_weight * batch_weight
                        entropy = entropy * mc_weight * batch_weight
                        y_LL = y_LL * mc_weight * batch_weight
                        prior_LL = prior_LL * mc_weight * batch_weight
                        v_LL = v_LL * mc_weight * batch_weight
                    loss_vals.append(loss.item())
                    entropy_vals.append(entropy.item())
                    prior_LL_vals.append(prior_LL.item())
                    y_LL_vals.append(y_LL.item())
                    v_LL_vals.append(v_LL.item())

                    loss.backward()

                    if not accumulate_gradient:
                        optimizer.step()
                        optimizer.zero_grad()

                    # prev_z = z_samples[..., -1].detach() # (n_mc_z, ntrials, b)
                    # prev_mu = mus[-1, ...].detach() # (ntrials, b)
                    # prev_Sigma = Sigmas[-1, ...].detach() # (ntrials, b, b)
                        
            if accumulate_gradient:
                optimizer.step()
                optimizer.zero_grad()
                
            scheduler.step()
            Z_y = self.gen_model.T * self.gen_model.ntrials * self.gen_model.N
            Z_z = self.gen_model.T * self.gen_model.ntrials * self.gen_model.b
            if v is not None:
                Z_v = self.gen_model.T * self.gen_model.ntrials * self.preprocessor.v_dim
            else:
                Z_v = 0
            self.LLs.append(-np.sum(loss_vals)/(Z_y + Z_v))
            self.entropy_vals.append(np.sum(entropy_vals)/(Z_z)) # TODO: this scaling may be incorrect
            self.y_LL_vals.append(np.sum(y_LL_vals)/(Z_y))
            self.prior_LL_vals.append(np.sum(prior_LL_vals)/(Z_z))
            if Z_v == 0:
                Z_v = 1
            self.v_LL_vals.append(np.sum(v_LL_vals)/(Z_v))
            if i % train_params_recognition['print_every'] == 0:
                print('step {} LL {:.4f} Entropy {:.4f} ln p(y|z) {:.4f} ln p(z) {:.4f} ln p(v|z) {:.4f}'.format(i, self.LLs[-1], self.entropy_vals[-1], self.y_LL_vals[-1], self.prior_LL_vals[-1], self.v_LL_vals[-1]))
                if train_params_recognition['print_deltas']:
                    print_str = ''
                    if len(self.dWs) > 0:
                        print_str += 'dWs: ' + str(self.dWs[-1])
                    if len(self.dRs) > 0:
                        print_str += 'dRs: ' + str(self.dRs[-1])
                    print(print_str)

    def LL(self, n_mc_x: int, x_hat:Tensor, y: Tensor, Ks: Tensor, Cs: Tensor, Sigmas_tilde_chol: Tensor, pseudo_obs=None, v=None):
        # y is (ntrials, N, batch_size)
        # x_hat is (n_mc_z, ntrials, x_dim, batch_size)

        # mus_smooth are the samples from the posterior through matheron sampling
        mus_filt, mus_smooth, mus_diffused = self.kalman_means(x_hat, Ks, Cs, pseudo_obs=pseudo_obs) # (T, n_mc_z, ntrials, b), (T, n_mc_z, ntrials, b), (T-1, n_mc_z, ntrials, b)
        entropy = self.entropy(mus_smooth, mus_filt, mus_diffused, Cs, Sigmas_tilde_chol) # (ntrials,)
        joint_LL, y_LL, prior_LL = self.gen_model.joint_LL(n_mc_x, mus_smooth.permute(1,2,3,0), y, prev_z=None, ret_only_joint=False) # (ntrials,) # TODO: batching
        if v is not None:
            v_LL = self.preprocessor.log_lik(mus_smooth.permute(1,2,3,0), v)
        else:
            v_LL = torch.zeros(joint_LL.shape).to(device)
        return (entropy + joint_LL + v_LL).sum(), entropy.sum(), y_LL.sum(), prior_LL.sum(), v_LL.sum()
    
    def freeze_params(self):
        for param in self.neural_net.parameters():
            param.requires_grad = False

    def test_z(self, test_y: Tensor):
        # TODO batching
        # return posterior mean on test data
        
        # test_y is (ntrials, N, T_test)
        pseudo_obs = self.get_x_tilde(test_y, only_x_tilde=False) # (ntrials, x_dim, T_test)
        x_tilde = pseudo_obs['x_tilde']
        _, _, Ks, Cs = self.kalman_covariance(T=test_y.shape[-1], pseudo_obs=pseudo_obs) # TODO: should I use prev_mu and prev_Sigma?
        _ , mus_smooth, _ = self.kalman_means(x_tilde[None, ...], Ks, Cs, pseudo_obs=pseudo_obs) # (T_test, 1, ntrials, b)
        mus_smooth = mus_smooth.squeeze(1) # (T_test, ntrials, b)
        return mus_smooth.permute(1, 2, 0) # (ntrials, b, T_test)
    
    def plot_LL(self):
        plt.plot(self.LLs)
        plt.xlabel('Step')
        plt.ylabel('LL')
        plt.show()
