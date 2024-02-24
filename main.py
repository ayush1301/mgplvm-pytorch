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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_symmetric(m: Tensor):
    return (m + m.T)/2

def is_symmetric(matrix):
    return torch.all(matrix == matrix.T)

def general_kalman_covariance(A, W, Q, R, b, x_dim, Sigma0, T=None, get_sigma_tilde=False, smoothing=True): # return all matrices independent of observations
    # Kalman filter
    Sigmas_filt = [] # Sigmas_filt[t] = Sigma_t^t 
    Sigmas_diffused = [] # Sigmas_diffused[t] = Sigma_{t+1}^t
    Sigmas_diffused_chol = torch.zeros(T - 1, b, b).to(device) # Sigmas_diffused_chol[t] = Cholesky of Sigmas_diffused[t]
    Sigmas_tilde = torch.zeros(T, b, b).to(device) # Sigmas_tilde[t] = Cov of p(z_t|z_{t+1:T}, y_{1:T})
    Ks = [] # Ks[t] = K_t

    # print if Q, R, Sigma0 are not symmetric
    if not (is_symmetric(Q) and is_symmetric(R) and is_symmetric(Sigma0)):
        print('assymetry in cov!!')

    # Sub in the first time step
    S = torch.linalg.cholesky(R + W @ Sigma0 @ W.T) # (x_dim, x_dim)
    K = chol_inv(S, Sigma0 @ W.T, left=False) # (b, x_dim)
    Sigmas_filt.append(make_symmetric(Sigma0 - K @ W @ Sigma0)) # (b, b)
    Ks.append(K)

    # Remaining time steps
    for t in range(1, T):
        # populate Sigmas_diffused[t-1] and Sigmas_filt[t]
        jitter = 1e-6 * torch.eye(b).to(device)
        Sigmas_diffused.append(make_symmetric(A @ Sigmas_filt[t-1] @ A.T + Q))
        Sigmas_diffused_chol[t-1] = torch.linalg.cholesky(Sigmas_diffused[t-1] + jitter) # (b, b)

        # K = Sigmas_diffused[t-1] @ W.T @ torch.linalg.inv(R + W @ Sigmas_diffused[t-1] @ W.T) # (b, x_dim)
        jitter = 1e-6 * torch.eye(x_dim).to(device)
        S = torch.linalg.cholesky(R + W @ Sigmas_diffused[t-1] @ W.T + jitter) # (x_dim, x_dim)
        K = chol_inv(S, Sigmas_diffused[t-1] @ W.T, left=False) # (b, x_dim)
        Sigmas_filt.append(make_symmetric(Sigmas_diffused[t-1] - K @ W @ Sigmas_diffused[t-1])) # (b, b)
        Ks.append(K)
    
    # Kalman smoother
    if smoothing:
        Sigmas_tilde[-1] = Sigmas_filt[-1] # (b, b)
        Cs = torch.zeros(T- 1, b, b).to(device) # Cs[t] = C_t
        for t in range(T - 2, -1, -1):
            # Cs[t] = Sigmas_filt[t] @ A.T @ torch.linalg.inv(Sigmas_diffused[t]) # (b, b)
            S = Sigmas_diffused_chol[t] # (b, b)
            Cs[t] = chol_inv(S, Sigmas_filt[t] @ A.T, left=False) # (b, b)
            Sigmas_tilde[t] = make_symmetric(Sigmas_filt[t] - Cs[t] @ Sigmas_diffused[t] @ Cs[t].T)
            # # print min eigenvalue of Sigmas_tilde[t]
            # print(torch.linalg.eigvals(Sigmas_tilde[t]))
        Sigmas_tilde_chol = torch.linalg.cholesky(Sigmas_tilde + 1e-4 * torch.eye(b).to(device)) # (b, b)

        # print(torch.linalg.det(Sigmas_tilde).mean())

        if get_sigma_tilde:
            return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks), Cs, Sigmas_tilde_chol
        else:
            return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks), Cs # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b)
    else:
        return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks) # (T, b, b), (T-1, b, b), (T, b, x_dim)
    
def general_kalman_means(A, W, b, mu0, x_hat: Tensor, Ks: Tensor, Cs: Tensor, smoothing=True):
    # TODO: support batching
    n_mc_z, n_trials, x_dim, batch_size = x_hat.shape
    _xhat = x_hat.permute(-1, 0, 1, 2) # (batch_size, n_mc_z, ntrials, x_dim)

    # Kalman filter
    mus_filt = [] # mu_filt[t] = mu_t^t # (batch_size, n_mc_z, ntrials, b)
    mus_diffused = [] # mu_diffused[t] = mu_{t+1}^t # (batch_size-1, n_mc_z, ntrials, b)

    # Sub in the first time step
    mus_filt.append(mu0 + (Ks[0] @ (_xhat[0] - W @ mu0)[..., None]).squeeze(-1)) # (n_mc_z, ntrials, b)

    start_t = 1

    for t in range(start_t, batch_size):
        # populate mus_diffused[t-1] and mus_filt[t]
        mus_diffused.append((A @ mus_filt[t-1][..., None]).squeeze(-1)) # (n_mc_z, ntrials, b)
        mus_filt.append(mus_diffused[t-1] + (Ks[t] @ (_xhat[t][..., None] - W @ mus_diffused[t-1][..., None])).squeeze(-1)) # (n_mc_z, ntrials, b)

    # Kalman smoother
    if smoothing:
        mus_smooth = torch.zeros(batch_size, n_mc_z, n_trials, b).to(device) # mu_smooth[t] = mu_t^T
        mus_smooth[-1] = mus_filt[-1] # (n_mc_z, ntrials, b)
        for t in range(batch_size - 2, -1, -1):
            mus_smooth[t] = mus_filt[t] + (Cs[t] @ (mus_smooth[t+1] - mus_diffused[t])[..., None]).squeeze(-1) # (n_mc_z, ntrials, b)
            
        return torch.stack(mus_filt), mus_smooth, torch.stack(mus_diffused) # (batch_size, n_mc_z, ntrials, b), (batch_size, n_mc_z, ntrials, b), (batch_size-1, n_mc_z, ntrials, b)
    else:
        return torch.stack(mus_filt), torch.stack(mus_diffused)
    
# returns (U @ U^T)^-1 @ x left=True, x @ (U @ U^T)^-1 left=False
def chol_inv(u, x, left=True):
    if left:
        # Solve U z = x
        u_inv_x = torch.linalg.solve_triangular(u, x, upper=False)
        # Solve U^T y = z
        return torch.linalg.solve_triangular(u.T, u_inv_x, upper=True)
    else:
        # Solve z U^T = x
        z_u_inv = torch.linalg.solve_triangular(u.T, x, upper=True, left=False)
        # Solve y U = z
        return torch.linalg.solve_triangular(u, z_u_inv, upper=False, left=False)

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
                 trained_z=False, d=0., fixed_d=True, single_sigma_x=False) -> None:
        super().__init__(z, Y, lik)

        self.link_fn = link_fn

        if x_dim is None:
            self.x_dim = self.b
        else:
            self.x_dim = x_dim

        # Generative parameters
        if A is None:
            # A = torch.rand(1, self.b, self.b).to(device)
            A = torch.eye(self.b)[None, ...].to(device)
        if C is None:
            # C = torch.randn(1, self.N, self.x_dim).to(device) / np.sqrt(self.x_dim)

            # Create an identity matrix of size x_dim
            identity = torch.eye(self.x_dim).to(device)

            # Repeat the identity matrix N//x_dim times
            C = identity.repeat(self.N//self.x_dim, 1)

            # If N is not a multiple of x_dim, append the remaining rows with zeros
            if self.N % self.x_dim != 0:
                zeros = torch.zeros((self.N % self.x_dim, self.x_dim)).to(device)
                C = torch.cat((C, zeros))

            # Reshape C to have shape (1, N, x_dim)
            C = C.unsqueeze(0)

        if W is None:
            # W = torch.randn(1, self.x_dim, self.b).to(device) / np.sqrt(self.b)

            # Create an identity matrix of size x_dim
            identity = torch.eye(self.x_dim).to(device)

            # Slice the identity matrix to get the first b columns
            W = identity[:, :self.b]

            # Reshape W to have shape (1, x_dim, b)
            W = W.unsqueeze(0)
        if B is None:
            # B = torch.rand(1, self.b, self.b).to(device)
            B = torch.eye(self.b)[None, ...].to(device)
        if mu0 is None:
            # mu0 = torch.rand(1, self.b).to(device)
            mu0 = torch.zeros(self.b).to(device)
        if Sigma0_half is None:
            Sigma0_half = 0.1 * torch.eye(self.b)[None, ...].to(device)
        if sigma_x is None:
            # sigma_x = torch.abs(torch.randn(1).to(device))
            # sigma_x = torch.tensor(0.1).to(device)
            sigma_x = 0.1 * torch.ones(self.x_dim).to(device)

        self.A = torch.nn.Parameter(A, requires_grad= not trained_z)
        self.C = torch.nn.Parameter(C)
        self.W = torch.nn.Parameter(W)
        self.B = torch.nn.Parameter(B, requires_grad= not trained_z)
        self.log_sigma_x = torch.nn.Parameter(torch.log(sigma_x))
        self.mu0 = torch.nn.Parameter(mu0, requires_grad= not trained_z)
        self.Sigma0_half = torch.nn.Parameter(Sigma0_half, requires_grad= not trained_z)
        self.d = torch.nn.Parameter(torch.tensor(d).to(device), requires_grad=not fixed_d)

        self.single_sigma_x = single_sigma_x

        # print(self.A.shape, self.B.shape, self.C.shape, self.sigma_x.shape, self.mu0.shape, self.Sigma0.shape)
    
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
        R = torch.diag(self.var_x).unsqueeze(0).to(device) # (1, x_dim, x_dim)
        jitter = torch.eye(R.shape[-1]).to(R.device) * 1e-6
        return R + jitter
    
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
        # samples = self.sigma_x * samples + mu[None, ...]
        samples = torch.diag(self.sigma_x) @ samples + mu[None, ...] # (n_mc, n_mc_z, ntrials, x_dim, T)
        # print(samples.shape)
        firing_rates = self.link_fn(C[None, ...] @ samples + self.d) # (n_mc_z, n_mc, ntrials, N, T)
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
    def __init__(self, gen_model: LDS, hidden_layer_size: int = 100, neural_net=None, rnn=False) -> None:
        super(RecognitionModel, self).__init__()
        self.gen_model = gen_model
        # Define a 2 layer MLP with hidden_layer_size hidden units
        if neural_net is None:
            if rnn:
                self.neural_net = RNNModel(gen_model.N, hidden_layer_size, gen_model.x_dim).to(device)
            else:
                self.neural_net = torch.nn.Sequential(
                    torch.nn.Linear(gen_model.N, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, gen_model.x_dim)
                ).to(device)
        else:
            self.neural_net = neural_net.to(device)
    
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
        }

        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print('adding', key)

        if params['batch_size'] is None:
            params['batch_size'] = self.gen_model.ntrials

        return params

    def train_full_model(self, train_params_gen, train_params_recognition):
        self.gen_model.train_supervised_model(train_params_gen)
        self.gen_model.freeze_params()
        self.train_recognition_model(train_params_recognition)
        
    def train_recognition_model(self, train_params_recognition):
        # dataset = Dataset(self.gen_model.Y)

        # So that the last dimension is the batch dimension
        # transposed_Y = self.gen_model.Y.transpose(0, -1) # (T, N, ntrials)
        # dataloader = DataLoader(transposed_Y, batch_size=train_params_recognition['batch_size'])
        dataloader = DataLoader(self.gen_model.Y, batch_size=train_params_recognition['batch_size'])

        self.fit(dataloader, train_params_recognition)
    
    def get_x_tilde(self, y: Tensor):
        # y is (ntrials, N, batch_size)
        x_tilde = self.neural_net(y.transpose(-1, -2)) # (ntrials, batch_size, x_dim)
        return x_tilde.transpose(-1, -2) # (ntrials, x_dim, batch_size)
    
    def sample_matheron_pert(self, n_mc: int, trials):
        z_prior = self.gen_model.sample_z(n_mc, trials).transpose(-1, -2) # (n_mc, ntrials, T, b) # TODO: how to deal with batch_size?
        pertubation = self.gen_model.W[None, ...] @ z_prior[..., None] # (n_mc, ntrials, T, x_dim, 1)
        noise = torch.linalg.cholesky(self.gen_model.R) @ torch.randn(*pertubation.shape).to(device) # (n_mc, ntrials, T, x_dim, 1)
        return (pertubation + noise).squeeze(-1).transpose(-1, -2) # (n_mc, ntrials, x_dim, T)
    
    def kalman_covariance(self, T=None, get_sigma_tilde=False): # return all matrices independent of observations
        if T is None:
            T = self.gen_model.T
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model.W.squeeze(0) # (x_dim, b)
        Q = self.gen_model.Q.squeeze(0) # (b, b)
        R = self.gen_model.R.squeeze(0)
        Sigma0 = self.gen_model.Sigma0.squeeze(0) # (b, b)
        b = self.gen_model.b
        x_dim = self.gen_model.x_dim
        return general_kalman_covariance(A, W, Q, R, b, x_dim, Sigma0, T, get_sigma_tilde)
    
    def kalman_means(self, x_hat: Tensor, Ks: Tensor, Cs: Tensor):
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model.W.squeeze(0) # (x_dim, b)
        b = self.gen_model.b
        mu0 = self.gen_model.mu0.squeeze(0) # (b)
        return general_kalman_means(A, W, b, mu0, x_hat, Ks, Cs)
    

    def entropy(self, samples, mus_filt, mus_diffused, Cs, Sigmas_tilde_chol):
        # samples is (batch_size, n_mc_z, ntrials, b)
        # mus_filt is (batch_size, n_mc_z, ntrials, b)
        # mus_diffused is (batch_size-1, n_mc_z, ntrials, b)
        # Cs is (batch_size-1, b, b)
        # Sigmas_tilde_chol is (batch_size, b, b)

        batch_size, n_mc_z, n_trials, b = samples.shape
        mus_tilde = torch.zeros(batch_size, n_mc_z, n_trials, b).to(device) # mu_tilde[t] = E[z_t|z_{t+1:T}, y_{1:T}]
        mus_tilde[-1] = mus_filt[-1] # (n_mc_z, ntrials, b)
        t_values = torch.arange(batch_size - 2, -1, -1)
        mus_tilde[t_values] = mus_filt[t_values] + (Cs[:, None, None, ...][t_values] @ (samples[t_values + 1] - mus_diffused[t_values])[..., None]).squeeze(-1)
        
        dist = MultivariateNormal(mus_tilde, scale_tril=Sigmas_tilde_chol[:, None, None, ...])
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
        optimizer = optimizer(self.neural_net.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params_recognition['step_size'], gamma=train_params_recognition['gamma'])

        if batch_mc_z is None:
            batch_mc_z = n_mc_z
        mc_batches = [batch_mc_z for _ in range(n_mc_z // batch_mc_z)]
        if (n_mc_z % batch_mc_z) > 0:
            mc_batches.append(n_mc_z % batch_mc_z)
        assert np.sum(mc_batches) == n_mc_z

        self.LLs = []
        _ , _, Ks, Cs, Sigmas_tilde_chol = self.kalman_covariance(get_sigma_tilde=True) # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b), (T, b, b)

        for i in range(max_steps):
            loss_vals = []
            prev_mu = None
            prev_Sigma = None
            prev_z = None
            for batch_mc_z in mc_batches:
                mc_weight = batch_mc_z / n_mc_z # fraction of the total samples
                for y in data: # loop over batches TODO
                    batch_trials = y.shape[0]
                    batch_weight = batch_trials / self.gen_model.ntrials
                    # NN pseudo observations
                    # y = y.transpose(-1, 0) # (ntrials, N, batch_size)
                    x_tilde = self.get_x_tilde(y) # (ntrials, x_dim, batch_size)

                    # Matheron psuedo observations
                    matheron_pert = self.sample_matheron_pert(batch_mc_z, batch_trials) # (n_mc_z, ntrials, x_dim, T) # TODO: do I need to do this every time?
                    x_hat = x_tilde[None, ...] - matheron_pert[..., :x_tilde.shape[-1]] # (n_mc_z, ntrials, x_dim, batch_size) TODO: how to deal with batch_size?
                    loss = -self.LL(n_mc_x, x_hat, y, Ks, Cs, Sigmas_tilde_chol).sum()

                    if accumulate_gradient:
                        loss = loss * mc_weight * batch_weight
                    loss_vals.append(loss.item())

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
            Z = self.gen_model.T * self.gen_model.ntrials * self.gen_model.N
            self.LLs.append(-np.sum(loss_vals)/(Z))
            if i % train_params_recognition['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])

    def LL(self, n_mc_x: int, x_hat:Tensor, y: Tensor, Ks: Tensor, Cs: Tensor, Sigmas_tilde_chol: Tensor):
        # y is (ntrials, N, batch_size)
        # x_hat is (n_mc_z, ntrials, x_dim, batch_size)

        # mus_smooth are the samples from the posterior through matheron sampling
        mus_filt, mus_smooth, mus_diffused = self.kalman_means(x_hat, Ks, Cs)
        entropy = self.entropy(mus_smooth, mus_filt, mus_diffused, Cs, Sigmas_tilde_chol) # (ntrials,)
        joint_LL = self.gen_model.joint_LL(n_mc_x, mus_smooth.permute(1,2,3,0), y, prev_z=None) # (ntrials,) # TODO: batching
        return entropy + joint_LL
    
    def freeze_params(self):
        for param in self.neural_net.parameters():
            param.requires_grad = False

    def test_z(self, test_y: Tensor):
        # TODO batching
        # return posterior mean on test data
        
        # test_y is (ntrials, N, T_test)
        x_tilde = self.get_x_tilde(test_y) # (ntrials, x_dim, T_test)
        _, _, Ks, Cs = self.kalman_covariance(T=test_y.shape[-1]) # TODO: should I use prev_mu and prev_Sigma?
        _ , mus_smooth, _ = self.kalman_means(x_tilde[None, ...], Ks, Cs) # (T_test, 1, ntrials, b)
        mus_smooth = mus_smooth.squeeze(1) # (T_test, ntrials, b)
        return mus_smooth.permute(1, 2, 0) # (ntrials, b, T_test)
    
    def plot_LL(self):
        plt.plot(self.LLs)
        plt.xlabel('Step')
        plt.ylabel('LL')
        plt.show()

class Preprocessor(Module):
    def __init__(self, v: Tensor, z_dim: int) -> None:
        # Generative parameters
        # z_t = A z_{t-1} + B w_t
        # v_t = W z_t + varepsilon_t

        super(Preprocessor, self).__init__()
        self.v = v.to(device) # (ntrials, v_dim, T)
        self.T = v.shape[-1]
        self.v_dim = v.shape[1]
        self.z_dim = z_dim # dimension of the latent space

        self.A = torch.nn.Parameter(torch.rand(self.z_dim, self.z_dim).to(device))
        self.B = torch.nn.Parameter(0.1 * torch.eye(self.z_dim).to(device))
        self.W = torch.nn.Parameter(torch.rand(self.v_dim, self.z_dim).to(device))
        self.mu0 = torch.nn.Parameter(torch.rand(self.z_dim).to(device))
        self.Sigma0_half = torch.nn.Parameter(0.1 * torch.eye(self.z_dim).to(device))
        self.log_sigma_v = torch.nn.Parameter(torch.log(torch.abs(torch.randn(1).to(device))))

    @property
    def sigma_v(self):
        return torch.exp(self.log_sigma_v)
    
    @property
    def var_v(self):
        return torch.square(self.sigma_v)
    
    @property
    def R(self):
        jitter = torch.eye(self.v_dim).to(self.sigma_v.device) * 1e-6
        return self.var_v * torch.eye(self.v_dim).to(device) + jitter
    
    @property
    def Q(self):
        self.B.data = torch.tril(self.B.data)
        jitter = torch.eye(self.z_dim).to(self.B.device) * 1e-6
        return self.B @ self.B.T + jitter
    
    @property
    def Sigma0(self):
        self.Sigma0_half.data = torch.tril(self.Sigma0_half.data)
        jitter = torch.eye(self.z_dim).to(self.Sigma0_half.device) * 1e-6
        return self.Sigma0_half @ self.Sigma0_half.T + jitter
    

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
        optimizer = optimizer(self.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params['step_size'], gamma=train_params['gamma'])

        self.LLs = []
        for i in range(max_steps):
            for v_transposed in data:
                optimizer.zero_grad()
                v = v_transposed.transpose(-1, 0) # (ntrials, v_dim, batch_size)
                Sigmas_diffused, Ks = self.kalman_covariance() # (T-1, z_dim, z_dim), (T, z_dim, v_dim)
                mus_diffused = self.kalman_means(v, Ks) # (T, ntrials, z_dim)
                loss = -self.LL(v, mus_diffused, Sigmas_diffused).mean()
                loss.backward()
                optimizer.step()
                self.LLs.append(-loss.item())
            scheduler.step()
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])

    def LL(self, v: Tensor, mus_diffused, Sigmas_diffused):
        # v is (ntrials, v_dim, T)
        # return log likelihood of v
        # mus_diffused is (T, ntrials, z_dim)
        # Sigmas_diffused is (T, z_dim, z_dim)
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
        jitter = torch.eye(self.v_dim).to(self.R.device) * 1e-6
        covs = self.W @ Sigmas_diffused[ts - 1] @ self.W.T + self.R + jitter # (T-1, v_dim, v_dim)
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