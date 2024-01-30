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
        dataset = MyDataset(self.z, self.Y)
        dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], collate_fn=MyDataset.my_collate_fn)

        self.fit(dataloader, train_params)

    def fit(self, data: DataLoader, train_params):
        lrate = train_params['lrate']
        n_mc = train_params['n_mc']
        max_steps = train_params['max_steps']

        optimizer = train_params['optimizer']
        optimizer = optimizer(self.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params['step_size'], gamma=train_params['gamma']) # TODO: understand these parameters

        LLs = []
        for i in range(max_steps):
            prev_z = None
            for z, y in data: # loop over batches
                loss = -self.joint_LL(n_mc, z, y, prev_z).mean() # TODO: should I use mean??
                loss.backward()
                LLs.append(-loss.item())
                optimizer.step()
                optimizer.zero_grad()
                prev_z = z[..., -1] # (ntrials, b)
            scheduler.step()
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', LLs[-1])
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False


class LDS(GenerativeModel):
    def __init__(self, z: Tensor, Y: Tensor, lik, x_dim=None, link_fn=torch.exp) -> None:
        super().__init__(z, Y, lik)

        self.link_fn = link_fn

        if x_dim is None:
            self.x_dim = self.b
        else:
            self.x_dim = x_dim

        # Generative parameters
        A = torch.rand(1, self.b, self.b).to(device)
        C = torch.rand(1, self.N, self.x_dim).to(device)
        W = torch.rand(1, self.x_dim, self.b).to(device)
        B = torch.rand(1, self.b, self.b).to(device)
        mu0 = torch.rand(1, self.b).to(device)
        Sigma0_half = 0.1 * torch.eye(self.b)[None, ...].to(device)
        sigma_x = torch.abs(torch.randn(1).to(device))

        self.A = torch.nn.Parameter(A)
        self.C = torch.nn.Parameter(C)
        self.W = torch.nn.Parameter(W)
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
    
    @property
    def R(self):
        R =  self.var_x * torch.eye(self.x_dim).unsqueeze(0).to(device) # (1, x_dim, x_dim)
        jitter = torch.eye(R.shape[-1]).to(R.device) * 1e-6
        return R + jitter
    
    # def joint_LL(self, n_mc, z, Y, 
    #              prev_z # (ntrials, b)
    #              ):
    #     # Natural parameters for p(x_t|v_t)
    #     ntrials, N, T = Y.shape # Only T is different from self.Y.shape
    #     _, b, _ = z.shape

    #     # Calculate first term of LL (p(y_{1:T}|z_{1:T}))
    #     mu = self.W @ z # (ntrials, x_dim, T)

    #     samples = torch.randn(n_mc, ntrials, self.x_dim, T).to(device)
    #     samples = self.sigma_x * samples + mu[None, ...]
    #     # print(samples.shape)
    #     firing_rates = self.link_fn(self.C[None, ...] @ samples) # (n_mc, ntrials, N, T)
    #     first = self.lik.LL(firing_rates, Y) # (ntrials,)
    #     # print(first.shape)

    #     ## Second term of LL p(z_{1:T})
    #     z0 = z[..., 0] # (ntrials, b)
    #     if prev_z is None:
    #         # print(self.mu0.shape, self.Sigma0.shape)
    #         second_small = MultivariateNormal(self.mu0, self.Sigma0).log_prob(z0) # (ntrials, )
    #     else:
    #         # p(z_0|z_{-1})
    #         mu = self.A @ prev_z[...,None] # (ntrials, b, 1)
    #         mu = mu[..., 0] # (ntrials, b)
    #         # print(mu.shape, self.Q.shape)
    #         second_small = MultivariateNormal(mu, self.Q).log_prob(z0) # (ntrials, ) # TODO: consider usong scale tril for speed
    #     # print(second_small.shape)

    #     # Natural parameters for p(z_t|z_{t-1})
    #     mus = self.A @ z[... , :-1] # (ntrials, b, T-1)
    #     # print(mu.shape, self.Q.shape)

    #     dist = MultivariateNormal(mus.transpose(-1, -2), self.Q[:, None, :, :])
    #     second_big = dist.log_prob(z[..., 1:].transpose(-1,-2)).sum(dim=-1)


    #     # print(second_big.shape)
    #     second = second_small + second_big

    #     return first + second
    

    def joint_LL(self, 
                 n_mc, # number of samples for x
                 z, # (ntrials, b, T) OR (n_mc_z, ntrials, b, T)
                 Y, # (ntrials, N, T)
                 prev_z # (ntrials, b) OR (n_mc_z, ntrials, b)
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
        samples = self.sigma_x * samples + mu[None, ...]
        # print(samples.shape)
        firing_rates = self.link_fn(C[None, ...] @ samples) # (n_mc_z, n_mc, ntrials, N, T)
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

        return first + second



class Poisson_noise():
    def __init__(self, d=0, fixed_d=True) -> None:
        self.d = torch.nn.Parameter(torch.tensor(d), requires_grad=not fixed_d)

    # def LL(self, rates, y) -> Tensor:
    #     '''
    #     x.shape = (n_mc, ntrials, N, T)
    #     y.shape = (ntrials, N, T)
    #     '''
    #     dist = Poisson(rates)
    #     log_prob = dist.log_prob(y[None, ...]) # (n_mc, ntrials, N, T)

    #     # avg_log_prob = torch.mean(log_prob, dim=0)  # (ntrials, N, T)
    #     # total_log_prob = torch.sum(avg_log_prob, dim=(1,2)) # (ntrials, )

    #     avg_log_prob = torch.logsumexp(log_prob, dim=0) - np.log(log_prob.shape[0]) # (ntrials, N, T)
    #     total_log_prob = torch.sum(avg_log_prob, dim=(1,2)) # (ntrials, )

    #     return total_log_prob
        
    def LL(self, rates, y) -> Tensor:
        '''
        x.shape = (n_mc_z, n_mc, ntrials, N, T)
        y.shape = (ntrials, N, T)
        '''
        dist = Poisson(rates + 1e-6) # TODO: is this a good idea? (adding small number to avoid log(0))
        log_prob = dist.log_prob(y[None, None, ...]) # (n_mc_z, n_mc, ntrials, N, T)

        avg_log_prob = torch.logsumexp(log_prob, dim=(0,1)) - np.log(log_prob.shape[0] * log_prob.shape[1]) # (ntrials, N, T)
        total_log_prob = torch.sum(avg_log_prob, dim=(-1, -2)) # (ntrials, )

        return total_log_prob

class Gaussian_noise():
    def __init__(self, sigma) -> None:
        self.sigma = sigma

class RecognitionModel(Module):
    def __init__(self, gen_model: LDS, hidden_layer_size: int = 100, smoothing: bool = False):
        super(RecognitionModel, self).__init__()
        self.gen_model = gen_model
        # Define a 2 layer MLP with hidden_layer_size hidden units
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(gen_model.N, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, gen_model.x_dim)
        ).to(device)
        self.smoothing = smoothing
    
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
            'batch_mc': None
        }

        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print('adding', key)

        if params['batch_size'] is None:
            params['batch_size'] = self.gen_model.T

        return params

    def train_full_model(self, train_params_gen, train_params_recognition):
        self.gen_model.train_supervised_model(train_params_gen)
        self.gen_model.freeze_params()
        self.train_recognition_model(train_params_recognition)
        
    def train_recognition_model(self, train_params_recognition):
        # dataset = Dataset(self.gen_model.Y)
        dataloader = DataLoader(self.gen_model.Y, batch_size=train_params_recognition['batch_size'])

        self.fit(dataloader, train_params_recognition)
    
    def get_x_tilde(self, y: Tensor):
        # y is (ntrials, N, batch_size)
        x_tilde = self.neural_net(y.transpose(-1, -2)) # (ntrials, batch_size, x_dim)
        return x_tilde.transpose(-1, -2) # (ntrials, x_dim, batch_size)
    
    def kalman_recursion(self, x_tilde: Tensor, prev_mu=None, prev_Sigma=None):
        # return Kalman filter mean and covariance
        n_trials, x_dim, batch_size = x_tilde.shape
        _xtilde = x_tilde.permute(2, 0, 1) # (batch_size, ntrials, x_dim)

        A = self.gen_model.A # (1, b, b)
        W = self.gen_model.W # (1, x_dim, b)
        Q = self.gen_model.Q # (1, b, b)
        R = self.gen_model.R # (1, x_dim, x_dim)

        # Kalman filter
        mus = torch.zeros(batch_size, n_trials, self.gen_model.b).to(device)
        Sigmas = torch.zeros(batch_size, n_trials, self.gen_model.b, self.gen_model.b).to(device)

        start_t = 0 # first index of mus and Sigmas to fill in
        if prev_mu is None or prev_Sigma is None:
            # Sub in the first time step
            mu0 = self.gen_model.mu0 # (1, b)
            Sigma0 = self.gen_model.Sigma0 # (1, b, b)
            mus[0] = mu0.expand_as(mus[0]) # (ntrials, b)
            Sigmas[0] = Sigma0.expand_as(Sigmas[0]) # (ntrials, b, b)
            start_t = 1

            prev_mu = mus[0] 
            prev_Sigma = Sigmas[0] 

        const = W.transpose(-1, -2) @ torch.linalg.inv(R) @ W
        for t in range(start_t, batch_size):
            # Calculate covariance
            precision = const + torch.linalg.inv(A @ prev_Sigma @ A.transpose(-1, -2) + Q)
            Sigma = torch.linalg.inv(precision)

            # Calculate mean
            kalman_gain = Sigma @ W.transpose(-1,-2) @ torch.linalg.inv(R) 
            prediction = A @ prev_mu[..., None] # (ntrials, b, 1)
            prediction_error = _xtilde[t].unsqueeze(-1) - W @ prediction # (ntrials, x_dim, 1)
            correction = kalman_gain @ prediction_error # (ntrials, b, 1)
            mu = (prediction + correction).squeeze(-1) # (ntrials, b)

            # Store
            mus[t] = mu
            Sigmas[t] = Sigma

            # Update
            prev_mu = mu
            prev_Sigma = Sigma

        if self.smoothing:
            # TODO: implement smoothing
            pass
            
        
        return mus, Sigmas
    
    def entropy(self, mus: Tensor, Sigmas: Tensor):
        # mus is (batch_size, ntrials, b)
        # Sigmas is (batch_size, ntrials, b, b)
        # TODO: check if this is correct
        jitter = torch.eye(mus.shape[-1]).to(device) * 1e-2 # TODO: very high jitter
        entropy = MultivariateNormal(mus, Sigmas + jitter).entropy() # (batch_size, ntrials)
        entropy = entropy.sum(dim=0) # sum over time
        return entropy # (ntrials,)

    def sample_z(self, n_mc: int, mus: Tensor, Sigmas: Tensor):
        # TODO
        # mus is (batch_size, ntrials, b)
        # Sigmas is (batch_size, ntrials, b, b)

        samples = torch.randn(n_mc, *mus.shape).to(device) # (n_mc, batch_size, ntrials, b)
        jitter = torch.eye(mus.shape[-1]).to(device) * 1e-2 # TODO: very high jitter
        Sigma_half = torch.linalg.cholesky(Sigmas + jitter) # (batch_size, ntrials, b, b)
        samples = mus[None, ...] + (Sigma_half[None, ...] @  samples[..., None]).squeeze(-1) # (n_mc, batch_size, ntrials, b)
        return samples.permute(0, 2, 3, 1) # (n_mc, ntrials, b, batch_size)

    def fit(self, data: DataLoader, train_params_recognition):
        lrate = train_params_recognition['lrate']
        n_mc_x = train_params_recognition['n_mc_x']
        n_mc_z = train_params_recognition['n_mc_z']
        max_steps = train_params_recognition['max_steps']

        optimizer = train_params_recognition['optimizer']
        optimizer = optimizer(self.neural_net.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params_recognition['step_size'], gamma=train_params_recognition['gamma'])

        LLs = []
        for i in range(max_steps):
            prev_mu = None
            prev_Sigma = None
            prev_z = None
            for y in data: # loop over batches
                x_tilde = self.get_x_tilde(y) # (ntrials, x_dim, batch_size)
                mus, Sigmas = self.kalman_recursion(x_tilde, prev_mu, prev_Sigma) # (batch_size, ntrials, b), (batch_size, ntrials, b, b)
                z_samples = self.sample_z(n_mc_z, mus, Sigmas) # (n_mc_z, ntrials, b, batch_size)
                loss = -self.LL(n_mc_x, z_samples, y, prev_z, mus, Sigmas).mean() # TODO: should I use mean??
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                prev_z = z_samples[..., -1] # (n_mc_z, ntrials, b)
            scheduler.step()
            LLs.append(-loss.item())
            if i % train_params_recognition['print_every'] == 0:
                print('step', i, 'LL', LLs[-1])

    def LL(self, n_mc_x: int, z_samples: Tensor, y: Tensor, prev_z: Tensor, mus: Tensor, Sigmas: Tensor):
        # y is (ntrials, N, batch_size)
        # prev_z is (n_mc_z, ntrials, b)
        entropy = self.entropy(mus, Sigmas) # (ntrials,)
        joint_LL = self.gen_model.joint_LL(n_mc_x, z_samples, y, prev_z) # (ntrials,)
        return entropy + joint_LL
    
    def freeze_params(self):
        for param in self.neural_net.parameters():
            param.requires_grad = False

    def test_z(self, test_y: Tensor):
        # return posterior mean on test data
        
        # test_y is (ntrials, N, T_test)
        x_tilde = self.get_x_tilde(test_y)
        mus, _ = self.kalman_recursion(x_tilde) # TODO: should I use prev_mu and prev_Sigma?
        return mus.permute(1, 2, 0) # (ntrials, b, T_test)


