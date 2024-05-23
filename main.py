import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import abc
from torch.distributions import MultivariateNormal, Poisson, NegativeBinomial, Normal, Bernoulli
from itertools import chain
import torch.distributions as dists
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import Module
from torch.optim.lr_scheduler import StepLR, LambdaLR
from preprocessor import Preprocessor
from utils import general_kalman_covariance, general_kalman_means
from enum import Enum
import dill
from sklearn.metrics import r2_score

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

class BATCHING(Enum):
    TIME = 1
    TRIALS = 2
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
            'batch_type': BATCHING.TRIALS,
            'save_every': None,
            'save_name': None,
        }

        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print('adding', key)

        if params['batch_size'] is None:
            if params['batch_type'] == BATCHING.TIME:
                params['batch_size'] = self.T
            elif params['batch_type'] == BATCHING.TRIALS:
                params['batch_size'] = self.ntrials


        return params

    def train_supervised_model(self, train_params):
        batching = train_params['batch_type']
        if batching == BATCHING.TIME:
            dataset = MyDataset(self.z, self.Y)
            dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], collate_fn=MyDataset.my_collate_fn, shuffle=False)
            self.fit(dataloader, train_params)
        elif batching == BATCHING.TRIALS:
            dataset = TensorDataset(self.z, self.Y)
            if self.ntrials % train_params['batch_size'] != 0:
                print('Warning: batch_size does not divide ntrials')
            dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=False)
            self.fit(dataloader, train_params)
        else:
            raise ValueError('Invalid batching type')

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
                # print(z.shape, y.shape)
                loss = -self.joint_LL(n_mc, z, y, prev_z).mean() # TODO: should I use mean??
                loss.backward()
                loss_vals.append(loss.item())
                if not train_params['accumulate_gradient']:
                    optimizer.step()
                    optimizer.zero_grad()
                if train_params['batch_type'] == BATCHING.TIME:
                    prev_z = z[..., -1] # (ntrials, b)
                loss_vals.append(loss.item())

            if train_params['accumulate_gradient']:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            Z = self.T * (self.N + self.b) # TODO: check this
            self.LLs.append(-np.mean(loss_vals)/Z)
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])
            if train_params['save_every'] is not None and i % train_params['save_every'] == 0:
                dill.dump(self, open(train_params['save_name'] + '.pkl', 'wb'))
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

        if C is None:
            C = torch.randn(1, self.N, self.x_dim).to(device) / np.sqrt(self.x_dim)

        if W is None:
            W = torch.randn(1, self.x_dim, self.b).to(device) / np.sqrt(self.b)

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
                 ret_only_joint=True,
                 CD_mask = None, # (ntrials, N, T)
                 CD_keep_prob = 1. # Probability of keeping a neuron
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

        samples = torch.linalg.cholesky(self.R.squeeze(0)) @ samples + mu[None, ...] # (n_mc, n_mc_z, ntrials, x_dim, T)

        # print(samples.shape)
        firing_rates = self.link_fn(C[None, ...] @ samples + self.d[:, None]) # (n_mc_z, n_mc, ntrials, N, T)
        first = self.lik.LL(firing_rates, Y, CD_mask, CD_keep_prob) # (ntrials,)
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
    
    def sample_z(self, n_mc: int, trials, prev_z=None, T=None): # Sample z from the prior of the LDS
        # prev_z is (n_mc, ntrials, b)
        if T is None:
            T = self.T

        samples = torch.zeros(n_mc, trials, self.b, T).to(device)
        start_t = 0
        if prev_z is None:
            # TODO: optimise cholesky?
            z0 = self.mu0[None, ...] + (torch.linalg.cholesky(self.Sigma0)[None, ...] @ torch.randn(n_mc, trials, self.b, 1).to(device)).squeeze(-1) # (n_mc, ntrials, b)
            samples[..., 0] = z0
            start_t = 1
            prev_z = z0
        for t in range(start_t, T):
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

    def general_LL(self, dist, y, CD_mask=None, CD_keep_prob=1.):
        # y.shape = (ntrials, N, T)
        log_prob = dist.log_prob(y[None, None, ...]) # (n_mc_z, n_mc, ntrials, N, T)

        # avg_log_prob = torch.logsumexp(log_prob, dim=(0,1)) - np.log(log_prob.shape[0] * log_prob.shape[1]) # (ntrials, N, T)
        # if CD_mask is not None:
        #     avg_log_prob = avg_log_prob * CD_mask
        #     # New code to adjust the y LL
        #     avg_log_prob = avg_log_prob * (1/(1-CD_keep_prob)) # Adjusting for the fact that we are only using a fraction of the neurons
        # total_log_prob = torch.sum(avg_log_prob, dim=(-1, -2))

        # Trying correct LL
        avg_log_prob = torch.logsumexp(log_prob, dim=(1)) - np.log(log_prob.shape[1]) # (n_mc_z, ntrials, N, T)
        avg_log_prob = avg_log_prob.mean(dim=0) # (ntrials, N, T)
        if CD_mask is not None:
            avg_log_prob = avg_log_prob * CD_mask
            # New code to adjust the y LL
            avg_log_prob = avg_log_prob * (1/(1-CD_keep_prob)) # Adjusting for the fact that we are only using a fraction of the neurons
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
    
    def LL(self, rates, y, CD_mask=None, CD_keep_prob=1.) -> Tensor:
        '''
        rates.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = NegativeBinomial(total_count=self.total_count[None, None, None, :, None], logits=rates)
        return self.general_LL(dist, y, CD_mask, CD_keep_prob)
    
    def dist_mean(self, rates: Tensor):
        # rates.shape = (n_mc, ntrials, N, T)
        total_count = self.total_count.detach().cpu()
        dist = NegativeBinomial(total_count=total_count[None, None, :, None], logits=rates)
        return dist.mean
        
class Poisson_noise(Noise):
    def __init__(self) -> None:
        super().__init__()
        
    def LL(self, rates, y, CD_mask=None, CD_keep_prob=1.) -> Tensor:
        '''
        rates.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = Poisson(rates + 1e-6) # TODO: is this a good idea? (adding small number to avoid log(0))
        return self.general_LL(dist, y, CD_mask, CD_keep_prob)
    
    def dist_mean(self, rates: Tensor):
        return rates

class Gaussian_noise(Noise):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.log_sigma = torch.nn.Parameter(torch.tensor(np.log(sigma)).to(device))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def LL(self, rates, y, CD_mask=None, CD_keep_prob=1.) -> Tensor:
        '''
        rates.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = Normal(rates, self.sigma)
        return self.general_LL(dist, y, CD_mask, CD_keep_prob)

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
    
class MyLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=True, dropout=0.):
        super(MyLSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            self.fc = torch.nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class RecognitionModel(Module):
    def __init__(self, gen_model: LDS, hidden_layer_size: int = 100, neural_net=None, rnn=False, cov_change=False, zero_mean_x_tilde=False, preprocessor: Preprocessor = None, gen_model_fixed=True, CD_keep_prob=1.,
                 Y_test: Tensor = None, v_test: np.ndarray = None, v_train: np.ndarray = None, held_out_neurons: np.ndarray = None, smoothing=True) -> None:
        super(RecognitionModel, self).__init__()
        self.gen_model = gen_model
        # Define a 2 layer MLP with hidden_layer_size hidden units
        if neural_net is None:
            if rnn:
                self.neural_net = MyLSTMModel(gen_model.N, hidden_layer_size, gen_model.x_dim).to(device)
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

        self.CD_keep_prob = CD_keep_prob # Probability for Coordinated Dropout

        self.Y_test = None
        if Y_test is not None:
            self.Y_test = Y_test.to(device)
        self.v_test = None
        if v_test is not None:
            self.v_test = v_test
        if v_train is not None:
            self.v_train = v_train

        self.test_neurons = held_out_neurons # list of neuron indeces that are completely held out
        self.train_neurons = None
        if held_out_neurons is not None:
            all_indices = set(range(self.gen_model.N))
            held_out_neurons_set = set(held_out_neurons)
            remaining_indices = np.array(sorted(list(all_indices - held_out_neurons_set)))
            self.train_neurons = remaining_indices
            # TODO: maybe assert here that len(train_neurons) == input size of the neural net
            # TODO: assertion for CD?

        self.smoothing = smoothing # Whether training is in the smoothing setting

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
            'save_every': None,
            'save_name': None,
            'train_co_smoothing_samps': 100,
            'train_co_smoothing_samps_per_batch': 10,
            'test_co_smoothing_samps': 100,
            'test_co_smoothing_samps_per_batch': 100
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
            dataloader = DataLoader(self.gen_model.Y, batch_size=train_params_recognition['batch_size'], shuffle=True)
        else:
            dataset = TensorDataset(self.gen_model.Y, self.preprocessor.v)
            dataloader = DataLoader(dataset, batch_size=train_params_recognition['batch_size'], shuffle=True)

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
    
    def sample_matheron_pert(self, n_mc: int, trials, pseudo_obs=None, T=None):
        z_prior = self.gen_model.sample_z(n_mc, trials, T=T).transpose(-1, -2) # (n_mc, ntrials, T, b) # TODO: how to deal with batch_size?
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
    
    def kalman_covariance(self, T=None, get_sigma_tilde=False, pseudo_obs=None, smoothing=True, filter_entropy_terms=False): # return all matrices independent of observations
        if T is None:
            T = self.gen_model.T
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model_W(pseudo_obs)
        Q = self.gen_model.Q.squeeze(0) # (b, b)
        R = self.gen_model_R(pseudo_obs)
        Sigma0 = self.gen_model.Sigma0.squeeze(0) # (b, b)
        b = self.gen_model.b
        x_dim = self.gen_model.x_dim
        return general_kalman_covariance(A, W, Q, R, b, x_dim, Sigma0, T, get_sigma_tilde, smoothing, filter_entropy_terms)
    
    def kalman_means(self, x_hat: Tensor, Ks: Tensor, Cs: Tensor, pseudo_obs=None, smoothing=True, filter_entropy_K=None):
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model_W(pseudo_obs)
        b = self.gen_model.b
        mu0 = self.gen_model.mu0.squeeze(0) # (b)
        return general_kalman_means(A, W, b, mu0, x_hat, Ks, Cs, smoothing, entropy_K=filter_entropy_K)
    

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
        # TODO THESE ARE THE SAME AS SAMPLES!!!!!
        mus_tilde[t_values] = mus_filt[t_values] + (Cs[:, None, ...][t_values] @ (samples[t_values + 1] - mus_diffused[t_values])[..., None]).squeeze(-1)
        
        dist = MultivariateNormal(mus_tilde, scale_tril=Sigmas_tilde_chol[:, None, ...])
        # dist = MultivariateNormal(mus_tilde, covariance_matrix=Sigmas_tilde[:, None, None, ...])
        log_prob = dist.log_prob(samples).sum(dim=0) # (n_mc_z, ntrials)
        return -log_prob.mean(dim=0) # (ntrials,)
    
    def entropy_filt(self, samples, filter_entropy_terms, mus_bar, x_tilde):
        # samples is (batch_size, n_mc_z, ntrials, b)
        # mus_bar is (batch_size-1, n_mc_z, ntrials, b)
        # x_tilde is (ntrials, x_dim, batch_size)
        Sigma_bar = filter_entropy_terms['Sigma_bar'] # (ntrials, b, b)
        Sigma_filt_first = filter_entropy_terms['Sigma_filt_first'] # (ntrials, b, b)

        # W = self.gen_model.W.squeeze(0) # (N, b)
        # A = self.gen_model.A.squeeze(0) # (b, b)
        # z_diffused = (A @ samples[:-1][..., None]) # (batch_size-1, n_mc_z, ntrials, b, 1)
        # K = filter_entropy_terms['K'].squeeze(0)
        # mus_bar = (z_diffused + K @ (x_tilde[..., 1:].permute(-1,0,1)[:, None, ...] - (W @ z_diffused).squeeze(-1))[..., None]).squeeze(-1) # (batch_size-1, n_mc_z, ntrials, b)
        

        # TODO Stupid way to calculate entropy
        first_dist = MultivariateNormal(samples[0], covariance_matrix=Sigma_filt_first)
        first_entropy = -first_dist.log_prob(samples[0]).mean(dim=0) # (ntrials,)

        # dist = MultivariateNormal(samples[1:,...], covariance_matrix=Sigma_bar)
        dist = MultivariateNormal(mus_bar, covariance_matrix=Sigma_bar)
        log_prob = dist.log_prob(samples[1:,...]).sum(dim=0) # (n_mc_z-1, ntrials)

        return -log_prob.mean(dim=0) + first_entropy # (ntrials,)

        # Sigma_filt = filter_entropy_terms['Sigma_filt'] # (batch_size, ntrials, b, b)
        # # print(Sigma_filt.shape, samples.shape)
        # dist = MultivariateNormal(samples, covariance_matrix=Sigma_filt[:, None, ...])
        # log_prob = dist.log_prob(samples).sum(dim=0) # (n_mc_z, ntrials)
        # return -log_prob.mean(dim=0) # (ntrials,)
    
        # # Sample from this distribution
        # # samples = torch.zeros(samples.shape).to(device) # TODO remove this dependency
        # batch_size, n_mc_z, n_trials, b = samples.shape
        # Q = self.gen_model.Q.squeeze(0) # (b, b)
        # W = self.gen_model.W.squeeze(0) # (N, b)
        # R = self.gen_model.R.squeeze(0) # (N, N)
        # K = Q @ W.T @ torch.linalg.inv(W @ Q @ W.T + R) # (b, N)
        # Sigma = Q - K @ W @ Q
        # A = self.gen_model.A.squeeze(0) # (b, b)
        # mu0 = self.gen_model.mu0.squeeze(0) # (b)
        # Sigma0 = self.gen_model.Sigma0.squeeze(0) # (b, b)

        # K1 = Sigma0 @ W.T @ torch.linalg.inv(W @ Sigma0 @ W.T + R)
        # mu1 = mu0 + (K1 @ (x_tilde[..., 0] - W @ mu0)[..., None]).squeeze(-1)
        # Sigma1 = Sigma0 - K1 @ W @ Sigma0
        # samples = []
        # samples.append(mu1 + (torch.linalg.cholesky(Sigma1) @ torch.randn(n_mc_z, n_trials, b).to(device)[..., None]).squeeze(-1))
        # for t in range(1, batch_size):
        #     z_diffused = (A @ samples[-1][..., None]) # (n_mc_z, ntrials, b, 1)
        #     # samples.append(A @ samples[t-1] + (Q @ torch.randn(samples[t].shape).to(device)[..., None]).squeeze(-1))
        #     mu = (z_diffused + K @ (x_tilde[..., t-1][None, ...] - (W @ z_diffused).squeeze(-1))[..., None]).squeeze(-1)
        #     samples.append(mu + (torch.linalg.cholesky(Sigma) @ torch.randn(n_mc_z, n_trials, b).to(device)[..., None]).squeeze(-1))
        # samples = torch.stack(samples, dim=0)
        # first_dist = MultivariateNormal(samples[0], covariance_matrix=Sigma1)
        # first_entropy = -first_dist.log_prob(samples[0]).mean(dim=0) # (ntrials,)
        # dist = MultivariateNormal(samples[1:,...], covariance_matrix=Sigma)
        # log_prob = dist.log_prob(samples[1:,...]).sum(dim=0) # (n_mc_z-1, ntrials)

        # return -log_prob.mean(dim=0) + first_entropy, samples



    def get_CD_mask(self, trials, N, T):
        # aa2236 below was the original code but here number of ones is not fixed
        # mask = torch.bernoulli(torch.full((trials, N, T), self.CD_keep_prob)).to(device)

        # aa2236 below is the new code where number of ones is fixed
        total_elements = trials * N * T
        num_ones = int(total_elements * self.CD_keep_prob)
        # Create a 1D tensor with the required number of ones
        tensor_1d = torch.cat((torch.ones(num_ones), torch.zeros(total_elements - num_ones)))
        # Shuffle the 1D tensor to randomly distribute the ones
        tensor_1d = tensor_1d[torch.randperm(total_elements)]
        # Reshape the 1D tensor back into a 3D tensor
        mask = tensor_1d.view(trials, N, T).to(device)

        return mask

    def CD(self, y, mask):
        if self.CD_keep_prob == 1:
            return y
        else:
            return y * mask / self.CD_keep_prob

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
        self.r2x_smooth, self.r2y_smooth, self.r2_smooth, self.r2x_filt, self.r2y_filt, self.r2_filt = [], [], [], [], [], []
        self.r2x_smooth_train, self.r2y_smooth_train, self.r2_smooth_train, self.r2x_filt_train, self.r2y_filt_train, self.r2_filt_train = [], [], [], [], [], []
        self.train_co_smoothing_vals, self.test_co_smoothing_vals = [], []
        if self.gen_model_fixed and not self.cov_change:
            if self.smoothing:
                _ , _, Ks, Cs, Sigmas_tilde_chol = self.kalman_covariance(get_sigma_tilde=True) # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b), (T, b, b)
                filter_entropy_relevant_terms = None
            else:
                Sigmas_filt, _, Ks, K, Sigma_bar = self.kalman_covariance(get_sigma_tilde=True, smoothing=False, filter_entropy_terms=True)
                filter_entropy_relevant_terms = {'Sigma_filt_first': Sigmas_filt[0], 'K': K, 'Sigma_bar': Sigma_bar, 'Sigma_filt': Sigmas_filt}
                # Sigmas_filt = None # Free up memory
                Cs = None
                Sigmas_tilde_chol = None

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

                    # aa2236 below only train y to be used
                    if self.train_neurons is not None:
                        train_y = y[:, self.train_neurons, :]
                    else:
                        train_y = y
                    CD_mask = self.get_CD_mask(*train_y.shape)
                    pseudo_obs = self.get_x_tilde(self.CD(train_y, CD_mask), only_x_tilde=False)
                    x_tilde = pseudo_obs['x_tilde'] # (ntrials, x_dim, batch_size)
                    # Matheron psuedo observations
                    matheron_pert = self.sample_matheron_pert(batch_mc_z, batch_trials, pseudo_obs) # (n_mc_z, ntrials, x_dim, T) # TODO: do I need to do this every time?
                    x_hat = x_tilde[None, ...] - matheron_pert[..., :x_tilde.shape[-1]] # (n_mc_z, ntrials, x_dim, batch_size) TODO: how to deal with batch_size?
                    if self.cov_change or not self.gen_model_fixed:
                        if self.smoothing:
                            _ , _, Ks, Cs, Sigmas_tilde_chol = self.kalman_covariance(get_sigma_tilde=True, pseudo_obs=pseudo_obs) # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b), (T, b, b)
                            filter_entropy_relevant_terms = None
                        else:
                            Sigmas_filt, _, Ks, K, Sigma_bar = self.kalman_covariance(get_sigma_tilde=True, pseudo_obs=pseudo_obs, smoothing=False, filter_entropy_terms=True)
                            filter_entropy_relevant_terms = {'Sigma_filt_first': Sigmas_filt[0], 'K': K, 'Sigma_bar': Sigma_bar, 'Sigma_filt': Sigmas_filt}
                            # Sigmas_filt = None # Free up memory
                            Cs = None
                            Sigmas_tilde_chol = None

                    if self.CD_keep_prob == 1:
                        CD_mask_complement = None
                    else:
                        CD_mask_complement = 1 - CD_mask
                        if self.train_neurons is not None:
                            _CD_mask_complement = torch.ones_like(y).to(device)
                            _CD_mask_complement[:, self.train_neurons, :] = CD_mask_complement
                            CD_mask_complement = _CD_mask_complement
                        # aa2236 - changed to below temporarily
                        # CD_mask_complement = torch.ones_like(CD_mask).to(device)
                    # aa2236 below the entire y is used since it is used to evaluate joint LL
                    loss, entropy, y_LL, prior_LL, v_LL = self.LL(n_mc_x, x_hat, y, Ks, Cs, Sigmas_tilde_chol, pseudo_obs=pseudo_obs, v=v, CD_mask_complement=CD_mask_complement, filter_entropy_terms=filter_entropy_relevant_terms, x_tilde=x_tilde)
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
            if train_params_recognition['save_every'] is not None and i % train_params_recognition['save_every'] == 0:
                dill.dump(self, open(train_params_recognition['save_name'] + '.pkl', 'wb'))
                if self.Y_test is not None:
                    print('Test Results:')
                    r2_smooth, r2_filt = self.test_r2(print_results=True, train_indices=self.train_neurons)
                    print()
                    self.r2x_smooth.append(r2_smooth[0])
                    self.r2y_smooth.append(r2_smooth[1])
                    self.r2_smooth.append(r2_smooth[2])
                    self.r2x_filt.append(r2_filt[0])
                    self.r2y_filt.append(r2_filt[1])
                    self.r2_filt.append(r2_filt[2])
                    if (self.smoothing and self.r2_smooth[-1] == max(self.r2_smooth)) or (not self.smoothing and self.r2_filt[-1] == max(self.r2_filt)):
                        dill.dump(self, open(train_params_recognition['save_name'] + '_best.pkl', 'wb'))
                # Compute train r^2
                print('Train Results')
                r2_smooth, r2_filt = self.test_r2(test_y=self.gen_model.Y, test_v=self.v_train, print_results=True, train_indices=self.train_neurons)
                print()
                self.r2x_smooth_train.append(r2_smooth[0])
                self.r2y_smooth_train.append(r2_smooth[1])
                self.r2_smooth_train.append(r2_smooth[2])
                self.r2x_filt_train.append(r2_filt[0])
                self.r2y_filt_train.append(r2_filt[1])
                self.r2_filt_train.append(r2_filt[2])

                if self.train_neurons is not None:
                    # co smoothing in this case

                    # train co smoothing
                    print('Train Cosmoothing: ', end='')
                    train_co_smoothing = self.complete_co_smoothing(test_y=self.gen_model.Y, smoothing=True, samples=train_params_recognition['train_co_smoothing_samps'], batch=train_params_recognition['train_co_smoothing_samps_per_batch'], train_indices=self.train_neurons, test_indices=self.test_neurons).item()
                    print(train_co_smoothing)
                    self.train_co_smoothing_vals.append(train_co_smoothing)

                    if train_params_recognition['test_co_smoothing_samps'] > 0:
                        print('Test Cosmoothing: ', end='')
                        test_co_smoothing = self.complete_co_smoothing(test_y=self.Y_test, smoothing=True, samples=train_params_recognition['test_co_smoothing_samps'], batch=train_params_recognition['test_co_smoothing_samps_per_batch'], train_indices=self.train_neurons, test_indices=self.test_neurons).item()
                        print(test_co_smoothing)
                        self.test_co_smoothing_vals.append(test_co_smoothing)
                    torch.cuda.empty_cache()

    def LL(self, n_mc_x: int, x_hat:Tensor, y: Tensor, Ks: Tensor, Cs: Tensor, Sigmas_tilde_chol: Tensor, pseudo_obs=None, v=None, CD_mask_complement: Tensor=None, filter_entropy_terms=None, x_tilde=None):
        # y is (ntrials, N, batch_size)
        # x_hat is (n_mc_z, ntrials, x_dim, batch_size)
        if self.smoothing:
            mus = self.kalman_means(x_hat, Ks, Cs, pseudo_obs=pseudo_obs, smoothing=self.smoothing)
            mus_filt, mus_smooth, mus_diffused = mus
            posterior_samps = mus_smooth
            entropy = self.entropy(mus_smooth, mus_filt, mus_diffused, Cs, Sigmas_tilde_chol) # (ntrials,)
        else:
            mus = self.kalman_means(x_hat, Ks, Cs, pseudo_obs=pseudo_obs, smoothing=self.smoothing, filter_entropy_K=filter_entropy_terms['K'])
            mus_filt, _ , mus_bar = mus
            posterior_samps = mus_filt
            # entropy, posterior_samps = self.entropy_filt(mus_filt, filter_entropy_terms, mus_bar, x_tilde=x_tilde) # (ntrials,)
            entropy = self.entropy_filt(mus_filt, filter_entropy_terms, mus_bar, x_tilde=x_tilde) # (ntrials,)


        if self.train_neurons is None:
            CD_keep_prob = self.CD_keep_prob
        else:
            total_train_times = y.shape[0] * y.shape[-1] * len(self.train_neurons)
            total_test_times = y.shape[0] * y.shape[-1] * len(self.test_neurons)
            CD_keep_prob = (self.CD_keep_prob * total_train_times) / (total_train_times + total_test_times)
        joint_LL, y_LL, prior_LL = self.gen_model.joint_LL(n_mc_x, posterior_samps.permute(1,2,3,0), y, prev_z=None, ret_only_joint=False, CD_mask=CD_mask_complement, CD_keep_prob=CD_keep_prob) # (ntrials,) # TODO: batching
        if v is not None:
            v_LL = self.preprocessor.log_lik(posterior_samps.permute(1,2,3,0), v)
        else:
            v_LL = torch.zeros(joint_LL.shape).to(device)
        return (entropy + joint_LL + v_LL).sum(), entropy.sum(), y_LL.sum(), prior_LL.sum(), v_LL.sum()
    
    def freeze_params(self):
        for param in self.neural_net.parameters():
            param.requires_grad = False

    def test_z(self, test_y: Tensor, smoothing=True, samples: int = 0, batch: int = -1, train_indices=None):
        # TODO batching
        # return posterior mean on test data

        if train_indices is not None:
            test_y = test_y[:, train_indices, :]
        
        # test_y is (ntrials, N, T_test)
        pseudo_obs = self.get_x_tilde(test_y, only_x_tilde=False) # (ntrials, x_dim, T_test)
        x_tilde = pseudo_obs['x_tilde']
        Covs = self.kalman_covariance(T=test_y.shape[-1], pseudo_obs=pseudo_obs, smoothing=smoothing)
        Ks = Covs[2]
        if smoothing:
            Cs = Covs[3]
        else:
            Cs = None
        mus = self.kalman_means(x_tilde[None, ...], Ks, Cs, pseudo_obs=pseudo_obs, smoothing=smoothing) 
        if smoothing:
            z = mus[1] # (T_test, 1, ntrials, b)
        else:
            z = mus[0] # (T_test, 1, ntrials, b)
        z = z.squeeze(1) # (T_test, ntrials, b)
        z = z.permute(1, 2, 0).detach().cpu().numpy() # (ntrials, b, T_test)

        # Sample from the posterior
        if samples:
            if batch == -1:
                batch = samples

            ret_samples = []
            while samples:
                if batch > samples:
                    batch = samples

                # n_mc_z is just samples
                matheron_pert = self.sample_matheron_pert(n_mc=batch, trials=test_y.shape[0], pseudo_obs=pseudo_obs, T=test_y.shape[-1]) # (n_mc_z, ntrials, x_dim, T)
                x_hat = x_tilde[None, ...] - matheron_pert[..., :x_tilde.shape[-1]] # (n_mc_z, ntrials, x_dim, batch_size)
                mus = self.kalman_means(x_hat, Ks, Cs, pseudo_obs=pseudo_obs, smoothing=smoothing)
                if smoothing:
                    z_samples = mus[1] # (T_test, n_mc_z, ntrials, b)
                else:
                    z_samples = mus[0] # (T_test, n_mc_z, ntrials, b)
                z_samples = z_samples.permute(1, 2, 3, 0) # (n_mc_z, ntrials, b, T_test)
                ret_samples.append(z_samples.detach().cpu().numpy())

                samples -= batch
            return z, np.vstack(ret_samples)
        else:    
            return z
    
    def test_r2(self, test_y: Tensor = None, test_v: Tensor = None, print_results=True, train_indices=None):
        assert self.preprocessor is not None
        if test_v is None:
            assert self.v_test is not None
            test_v = self.v_test

        if test_y is None:
            if self.Y_test is None:
                raise ValueError('No test data provided')
            else:
                test_y = self.Y_test

        assert (test_y.shape[0], test_y.shape[-1]) == (test_v.shape[0], test_v.shape[-1])      

        z_smooth = self.test_z(test_y, train_indices=train_indices) # (ntrials, b, T_test)
        z_filt = self.test_z(test_y, smoothing=False, train_indices=train_indices) # (ntrials, b, T_test)

        v_smooth = self.preprocessor.W.detach().cpu().numpy() @ z_smooth # (ntrials, v_dim, T_test)
        v_filt = self.preprocessor.W.detach().cpu().numpy() @ z_filt # (ntrials, v_dim, T_test)

        r2x_smooth = r2_score(test_v[:,0,:].flatten(), v_smooth[:,0,:].flatten())
        r2y_smooth = r2_score(test_v[:,1,:].flatten(), v_smooth[:,1,:].flatten())
        r2x_filt = r2_score(test_v[:,0,:].flatten(), v_filt[:,0,:].flatten())
        r2y_filt = r2_score(test_v[:,1,:].flatten(), v_filt[:,1,:].flatten())
        r2_smooth = (r2x_smooth + r2y_smooth)/2
        r2_filt = (r2x_filt + r2y_filt)/2

        if print_results:
            print('R2 smooth: x = {:.4f}, y = {:.4f}, avg = {:.4f}'.format(r2x_smooth, r2y_smooth, r2_smooth))
            print('R2 filt: x = {:.4f}, y = {:.4f}, avg = {:.4f}'.format(r2x_filt, r2y_filt, r2_filt))
        return [r2x_smooth, r2y_smooth, r2_smooth], [r2x_filt, r2y_filt, r2_filt]

    def plot_LL(self):
        plt.plot(self.LLs)
        plt.xlabel('Step')
        plt.ylabel('LL')
        plt.show()

    def get_firing_rates(self, z_samps: np.ndarray):
        # z_samps is (n_mc, ntrials, z_dim, T)
        W = self.gen_model.W[0].detach().cpu()
        C = self.gen_model.C.detach().cpu()
        if len(C.shape) == 3: # For buggy identity C in some tests. Length should always be 3
            C = C[0]
        R = self.gen_model.R[0].detach().cpu()
        d = self.gen_model.d.detach().cpu()
        # print(W.shape, C.shape, R.shape, d.shape)
        X = W @ z_samps
        X += torch.linalg.cholesky(R) @ torch.randn(*X.shape)
        # print(C.shape, 'C shape')
        # print((C @ X ).shape, 'CX shape')
        # print((C @ X + d[:, None]).shape, 'cx + d shape')
        F_samps= self.gen_model.link_fn(C @ X + d[:, None])

        F = self.gen_model.lik.dist_mean(F_samps)
        return F.mean(dim=0), F_samps, X

    def co_smoothing(self, Y: Tensor, rates: Tensor):
        dist = Poisson(rate=rates)
        log_prob = dist.log_prob(Y)
        return log_prob.mean(), log_prob
    
    def complete_co_smoothing(self, test_y, smoothing, samples, batch, train_indices, test_indices):
        _, z_samps = self.test_z(test_y=test_y, smoothing=smoothing, samples=samples, batch=batch, train_indices=train_indices)
        torch.cuda.empty_cache()
        F = self.get_firing_rates(z_samps)[0]
        return self.co_smoothing(test_y[:, test_indices, :], F[:, test_indices, :].to(device))[0]