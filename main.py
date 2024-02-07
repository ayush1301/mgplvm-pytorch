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
        dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], collate_fn=MyDataset.my_collate_fn, shuffle=False)

        self.fit(dataloader, train_params)

    def fit(self, data: DataLoader, train_params):
        lrate = train_params['lrate']
        n_mc = train_params['n_mc']
        max_steps = train_params['max_steps']

        optimizer = train_params['optimizer']
        optimizer = optimizer(self.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params['step_size'], gamma=train_params['gamma']) # TODO: understand these parameters

        self.LLs = []
        for i in range(max_steps):
            prev_z = None
            for z, y in data: # loop over batches
                loss = -self.joint_LL(n_mc, z, y, prev_z).mean() # TODO: should I use mean??
                loss.backward()
                self.LLs.append(-loss.item())
                optimizer.step()
                optimizer.zero_grad()
                prev_z = z[..., -1] # (ntrials, b)
            scheduler.step()
            if i % train_params['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def plot_LL(self):
        plt.plot(self.LLs)
        plt.xlabel('Step')
        plt.ylabel('LL')
        plt.show()


class LDS(GenerativeModel):
    def __init__(self, z: Tensor, Y: Tensor, lik, x_dim=None, link_fn=torch.exp,
                 A=None, C=None, W=None, B=None, mu0=None, Sigma0_half=None, sigma_x=None,
                 trained_z=False) -> None:
        super().__init__(z, Y, lik)

        self.link_fn = link_fn

        if x_dim is None:
            self.x_dim = self.b
        else:
            self.x_dim = x_dim

        # Generative parameters
        if A is None:
            A = torch.rand(1, self.b, self.b).to(device)
        if C is None:
            C = torch.rand(1, self.N, self.x_dim).to(device)
        if W is None:
            W = torch.rand(1, self.x_dim, self.b).to(device)
        if B is None:
            B = torch.rand(1, self.b, self.b).to(device)
        if mu0 is None:
            mu0 = torch.rand(1, self.b).to(device)
        if Sigma0_half is None:
            Sigma0_half = 0.1 * torch.eye(self.b)[None, ...].to(device)
        if sigma_x is None:
            sigma_x = torch.abs(torch.randn(1).to(device))

        self.A = torch.nn.Parameter(A, requires_grad= not trained_z)
        self.C = torch.nn.Parameter(C)
        self.W = torch.nn.Parameter(W)
        self.B = torch.nn.Parameter(B, requires_grad= not trained_z)
        self.sigma_x = torch.nn.Parameter(sigma_x)
        self.mu0 = torch.nn.Parameter(mu0, requires_grad= not trained_z)
        self.Sigma0_half = torch.nn.Parameter(Sigma0_half, requires_grad= not trained_z)

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
    
    def sample_z(self, n_mc: int, prev_z=None): # Sample z from the prior of the LDS
        # prev_z is (n_mc, ntrials, b)
        samples = torch.zeros(n_mc, self.ntrials, self.b, self.T).to(device)
        start_t = 0
        if prev_z is None:
            z0 = self.mu0[None, ...] + (self.Sigma0_half[None, ...] @ torch.randn(n_mc, self.ntrials, self.b, 1).to(device)).squeeze(-1) # (n_mc, ntrials, b)
            samples[..., 0] = z0
            start_t = 1
            prev_z = z0
        for t in range(start_t, self.T):
            z_t = (self.A[None, ...] @ prev_z[..., None] + self.B[None, ...] @ torch.randn(n_mc, self.ntrials, self.b, 1).to(device)).squeeze(-1)
            samples[..., t] = z_t
            prev_z = z_t

        return samples



class Poisson_noise():
    def __init__(self, d=0, fixed_d=True) -> None:
        self.d = torch.nn.Parameter(torch.tensor(d), requires_grad=not fixed_d)
        
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
        self.smoothing = smoothing # TODO: implement smoothing
    
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

        # So that the last dimension is the batch dimension
        transposed_Y = self.gen_model.Y.transpose(0, -1) # (T, N, ntrials)
        dataloader = DataLoader(transposed_Y, batch_size=train_params_recognition['batch_size'])

        self.fit(dataloader, train_params_recognition)
    
    def get_x_tilde(self, y: Tensor):
        # y is (ntrials, N, batch_size)
        x_tilde = self.neural_net(y.transpose(-1, -2)) # (ntrials, batch_size, x_dim)
        return x_tilde.transpose(-1, -2) # (ntrials, x_dim, batch_size)
    
    def sample_matheron_pert(self, n_mc: int):
        z_prior = self.gen_model.sample_z(n_mc).transpose(-1, -2) # (n_mc, ntrials, T, b) # TODO: how to deal with batch_size?
        pertubation = self.gen_model.W[None, ...] @ z_prior[..., None] # (n_mc, ntrials, T, x_dim, 1)
        noise = torch.linalg.cholesky(self.gen_model.R) @ torch.randn(*pertubation.shape).to(device) # (n_mc, ntrials, T, x_dim, 1)
        return (pertubation + noise).squeeze(-1).transpose(-1, -2) # (n_mc, ntrials, x_dim, T)
    
    def kalman_covariance(self, T=None): # return all matrices independent of observations
        if T is None:
            T = self.gen_model.T
        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model.W.squeeze(0) # (x_dim, b)
        Q = self.gen_model.Q.squeeze(0) # (b, b)
        R = self.gen_model.R.squeeze(0) # (x_dim, x_dim)

        # Kalman filter
        Sigmas_filt = torch.zeros(T, self.gen_model.b, self.gen_model.b).to(device) # Sigmas_filt[t] = Sigma_t^t
        Sigmas_diffused = torch.zeros(T - 1, self.gen_model.b, self.gen_model.b).to(device) # Sigmas_diffused[t] = Sigma_{t+1}^t
        Ks = torch.zeros(T, self.gen_model.b, self.gen_model.x_dim).to(device) # Ks[t] = K_t

        # Sub in the first time step
        Sigma0 = self.gen_model.Sigma0.squeeze(0) # (b, b)
        K = Sigma0 @ W.T @ torch.linalg.inv(R + W @ Sigma0 @ W.T) # (b, x_dim)
        Sigmas_filt[0] = Sigma0 - K @ W @ Sigma0 # (b, b)
        Ks[0] = K

        # Remaining time steps
        for t in range(1, T):
            Sigmas_diffused[t-1] = A @ Sigmas_filt[t-1] @ A.T + Q # (b, b)
            K = Sigmas_diffused[t-1] @ W.T @ torch.linalg.inv(R + W @ Sigmas_diffused[t-1] @ W.T) # (b, x_dim)
            Sigmas_filt[t] = Sigmas_diffused[t-1] - K @ W @ Sigmas_diffused[t-1] # (b, b)
            Ks[t] = K
        
        # Kalman smoother
        Cs = torch.zeros(T- 1, self.gen_model.b, self.gen_model.b).to(device) # Cs[t] = C_t
        for t in range(T - 2, -1, -1):
            Cs[t] = Sigmas_filt[t] @ A.T @ torch.linalg.inv(Sigmas_diffused[t]) # (b, b)

        return Sigmas_filt, Sigmas_diffused, Ks, Cs # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b)


    def kalman_means(self, x_hat: Tensor, Ks: Tensor, Cs: Tensor):
        # TODO: support batching
        n_mc_z, n_trials, x_dim, batch_size = x_hat.shape
        _xhat = x_hat.permute(-1, 0, 1, 2) # (batch_size, n_mc_z, ntrials, x_dim)

        A = self.gen_model.A.squeeze(0) # (b, b)
        W = self.gen_model.W.squeeze(0) # (x_dim, b)

        # Kalman filter
        mus_filt = torch.zeros(batch_size, n_mc_z, n_trials, self.gen_model.b).to(device) # mu_filt[t] = mu_t^t
        mus_diffused = torch.zeros(batch_size-1, n_mc_z, n_trials, self.gen_model.b).to(device) # mu_diffused[t] = mu_{t+1}^t

        # Sub in the first time step
        mu0 = self.gen_model.mu0.squeeze(0) # (b)
        mus_filt[0] = mu0 + (Ks[0] @ (_xhat[0] - W @ mu0)[..., None]).squeeze(-1) # (n_mc_z, ntrials, b)

        start_t = 1

        for t in range(start_t, batch_size):
            mus_diffused[t-1] = (A @ mus_filt[t-1][..., None]).squeeze(-1) # (n_mc_z, ntrials, b)
            mus_filt[t] = mus_diffused[t-1] + (Ks[t] @ (_xhat[t][..., None] - W @ mus_diffused[t-1][..., None])).squeeze(-1) # (n_mc_z, ntrials, b)

        # Kalman smoother
        mus_smooth = torch.zeros(batch_size, n_mc_z, n_trials, self.gen_model.b).to(device) # mu_smooth[t] = mu_t^T
        mus_smooth[-1] = mus_filt[-1] # (n_mc_z, ntrials, b)
        for t in range(batch_size - 2, -1, -1):
            mus_smooth[t] = mus_filt[t] + (Cs[t] @ (mus_smooth[t+1] - mus_diffused[t])[..., None]).squeeze(-1) # (n_mc_z, ntrials, b)
            
        return mus_filt, mus_smooth, mus_diffused # (batch_size, n_mc_z, ntrials, b), (batch_size, n_mc_z, ntrials, b), (batch_size-1, n_mc_z, ntrials, b)
    
    def entropy(self, samples, mus_filt, Sigmas_filt, mus_diffused, Sigmas_diffused):
        # samples is (batch_size, n_mc_z, ntrials, b)
        # mus_filt is (batch_size, n_mc_z, ntrials, b)
        # Sigmas_filt is (batch_size, b, b)
        # mus_diffused is (batch_size-1, n_mc_z, ntrials, b)
        # Sigmas_diffused is (batch_size-1, b, b)

        # Add jitter to the covariance matrices
        Sigmas_filt = Sigmas_filt + 1E-6 * torch.eye(Sigmas_filt.shape[-1]).to(device)
        Sigmas_diffused = Sigmas_diffused + 1E-6 * torch.eye(Sigmas_diffused.shape[-1]).to(device)
        
        filter_dist = MultivariateNormal(mus_filt, covariance_matrix=Sigmas_filt[:, None, None, ...])
        filter_log_prob = filter_dist.log_prob(samples).sum(dim=0) # (n_mc_z, ntrials)

        diffused_dist = MultivariateNormal(mus_diffused, covariance_matrix=Sigmas_diffused[:, None, None, ...])
        diffused_log_prob = diffused_dist.log_prob(samples[1:, ...]).sum(dim=0) # (n_mc_z, ntrials)

        # TODO: this code is repeated computation
        # Natural parameters for p(z_t|z_{t-1})
        A = self.gen_model.A.squeeze(0) # (b, b)
        Q = self.gen_model.Q.squeeze(0) # (b, b)
        mus = (A @ samples[:-1, ..., None]).squeeze(-1) # (batch_size-1, n_mc_z, ntrials, b)
        prior_dist = MultivariateNormal(mus, covariance_matrix=Q)
        prior_log_prob = prior_dist.log_prob(samples[1:, ...]).sum(dim=0) # (n_mc_z, ntrials)

        log_prob = filter_log_prob - diffused_log_prob + prior_log_prob # (n_mc_z, ntrials)
        return -log_prob.mean(dim=0) # (ntrials,) 
      

    def fit(self, data: DataLoader, train_params_recognition):
        lrate = train_params_recognition['lrate']
        n_mc_x = train_params_recognition['n_mc_x']
        n_mc_z = train_params_recognition['n_mc_z']
        max_steps = train_params_recognition['max_steps']

        optimizer = train_params_recognition['optimizer']
        optimizer = optimizer(self.neural_net.parameters(), lr=lrate)
        scheduler = StepLR(optimizer, step_size=train_params_recognition['step_size'], gamma=train_params_recognition['gamma'])

        self.LLs = []
        Sigmas_filt, Sigmas_diffused, Ks, Cs = self.kalman_covariance() # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b)

        for i in range(max_steps):
            prev_mu = None
            prev_Sigma = None
            prev_z = None
            for y in data: # loop over batches
                # NN pseudo observations
                y = y.transpose(-1, 0) # (ntrials, N, batch_size)
                x_tilde = self.get_x_tilde(y) # (ntrials, x_dim, batch_size)
                # Matheron psuedo observations
                matheron_pert = self.sample_matheron_pert(n_mc_z) # (n_mc_z, ntrials, x_dim, T) # TODO: do I need to do this every time?
                x_hat = x_tilde[None, ...] - matheron_pert[..., :x_tilde.shape[-1]] # (n_mc_z, ntrials, x_dim, batch_size) TODO: how to deal with batch_size?
                loss = -self.LL(n_mc_x, x_hat, y, Sigmas_filt, Sigmas_diffused, Ks, Cs).mean() # TODO: should I use mean??
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # prev_z = z_samples[..., -1].detach() # (n_mc_z, ntrials, b)
                # prev_mu = mus[-1, ...].detach() # (ntrials, b)
                # prev_Sigma = Sigmas[-1, ...].detach() # (ntrials, b, b)
            scheduler.step()
            self.LLs.append(-loss.item())
            if i % train_params_recognition['print_every'] == 0:
                print('step', i, 'LL', self.LLs[-1])

    def LL(self, n_mc_x: int, x_hat:Tensor, y: Tensor, Sigmas_filt: Tensor, Sigmas_diffused: Tensor, Ks: Tensor, Cs: Tensor):
        # y is (ntrials, N, batch_size)
        # x_hat is (n_mc_z, ntrials, x_dim, batch_size)

        # mus_smooth are the samples from the posterior through matheron sampling
        mus_filt, mus_smooth, mus_diffused = self.kalman_means(x_hat, Ks, Cs)
        entropy = self.entropy(mus_smooth, mus_filt, Sigmas_filt, mus_diffused, Sigmas_diffused) # (ntrials,)
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

