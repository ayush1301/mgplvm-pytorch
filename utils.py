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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_symmetric(m: Tensor):
    new = (m + m.transpose(-1,-2))/2
    if not torch.allclose(new, m, atol=1e-6):
        print(torch.max(torch.abs(new - m)))
        raise ValueError('Matrix is not symmetric')
    return new

def is_symmetric(matrix):
    return torch.all(matrix == matrix.transpose(-1, -2))

def general_kalman_covariance(A, W, Q, R, b, x_dim, Sigma0, T=None, get_sigma_tilde=False, smoothing=True, ret_smoothing_cov=False): # return all matrices independent of observations
    W_var, R_var, Q_var, A_var = True, True, True, True
    if len(W.shape) == 2:
        W = W[None, None, ...] # T, ntrials, x_dim, b
        W_var = False
    if len(R.shape) == 2:
        R = R[None, None, ...] # T, ntrials, x_dim, x_dim
        R_var = False
    if len(A.shape) == 2:
        A = A[None, None, ...] # T-1, ntrials, b, b
        A_var = False
    if len(Q.shape) == 2:
        Q = Q[None, None, ...] # T-1, ntrials, b, b
        Q_var = False
    if len(Sigma0.shape) == 2:
        Sigma0 = Sigma0[None, ...] # ntrials, b, b
    trials = max(A.shape[1], W.shape[1], Q.shape[1], R.shape[1], Sigma0.shape[0])

    # Kalman filter
    Sigmas_filt = [] # Sigmas_filt[t] = Sigma_t^t 
    Sigmas_diffused = [] # Sigmas_diffused[t] = Sigma_{t+1}^t
    Sigmas_diffused_chol = [] # Sigmas_diffused_chol[t] = Cholesky of Sigmas_diffused[t], shape = T - 1, b, b at end
    Sigmas_tilde = [torch.zeros(trials, b, b).to(device) for _ in range(T)] # Sigmas_tilde[t] = Cov of p(z_t|z_{t+1:T}, y_{1:T})
    Ks = [] # Ks[t] = K_t
    if ret_smoothing_cov:
        Sigmas_smooth = [] # Sigmas_smooth[t] = Cov of p(z_t|y_{1:T})

    # print if Q, R, Sigma0 are not symmetric
    if not (is_symmetric(Q) and is_symmetric(R) and is_symmetric(Sigma0)):
        print('assymetry in cov!!')

    # Sub in the first time step
    S = torch.linalg.cholesky(R[0] + W[0] @ Sigma0 @ W[0].transpose(-1, -2)) # (ntrials, x_dim, x_dim)
    K = chol_inv(S, Sigma0 @ W[0].transpose(-1, -2), left=False) # (ntrials, b, x_dim)
    Sigmas_filt.append(make_symmetric(Sigma0 - K @ W[0] @ Sigma0)) # (ntrials, b, b)
    Ks.append(K)

    # Remaining time steps
    for t in range(1, T):
        # Find matrices relevant for this time step
        _A = A[t-1] if A_var else A[0]
        _W = W[t] if W_var else W[0]
        _Q = Q[t-1] if Q_var else Q[0]
        _R = R[t] if R_var else R[0]
    
        # populate Sigmas_diffused[t-1] and Sigmas_filt[t]
        jitter = 1e-6 * torch.eye(b).to(device)
        Sigmas_diffused.append(make_symmetric(_A @ Sigmas_filt[t-1] @ _A.transpose(-1, -2) + _Q)) # (ntrials, b, b)
        Sigmas_diffused_chol.append(torch.linalg.cholesky(Sigmas_diffused[t-1] + jitter))

        jitter = 1e-6 * torch.eye(x_dim).to(device)
        S = torch.linalg.cholesky(_R + _W @ Sigmas_diffused[t-1] @ _W.transpose(-1,-2) + jitter) # (ntrials, x_dim, x_dim)
        K = chol_inv(S, Sigmas_diffused[t-1] @ _W.transpose(-1,-2), left=False) # (ntrials, b, x_dim)
        Sigmas_filt.append(make_symmetric(Sigmas_diffused[t-1] - K @ _W @ Sigmas_diffused[t-1])) # (ntrials, b, b)
        Ks.append(K)
    
    # Kalman smoother
    if smoothing:
        Sigmas_tilde[-1] = Sigmas_filt[-1] # (b, b)
        Cs = [torch.zeros(trials, b, b).to(device) for _ in range(T - 1)] # Cs[t] = C_t
        if ret_smoothing_cov:
            Sigmas_smooth.append(Sigmas_filt[-1])
        for t in range(T - 2, -1, -1):
            _A = A[t-1] if A_var else A[0]
            _W = W[t] if W_var else W[0]
            _Q = Q[t-1] if Q_var else Q[0]
            _R = R[t] if R_var else R[0]
            S = Sigmas_diffused_chol[t] # (ntrials, b, b)
            Cs[t] = chol_inv(S, Sigmas_filt[t] @ _A.transpose(-1,-2), left=False) # (ntrials, b, b)
            Sigmas_tilde[t] = make_symmetric(Sigmas_filt[t] - Cs[t] @ Sigmas_diffused[t] @ Cs[t].transpose(-1,-2))
            if ret_smoothing_cov:
                Sigmas_smooth.append(Sigmas_filt[t] + Cs[t] @ (Sigmas_smooth[-1] - Sigmas_diffused[t]) @ Cs[t].transpose(-1,-2))
        # Find minimum eigenvalue of Sigmas_tilde
        print(torch.linalg.eigvalsh(torch.stack(Sigmas_tilde)).min())
        Sigmas_tilde_chol = torch.linalg.cholesky(torch.stack(Sigmas_tilde) + 1e-4 * torch.eye(b).to(device)) # (ntrials, b, b)

        # print(torch.linalg.det(Sigmas_tilde).mean())
        
        if ret_smoothing_cov: # return everything when ret_smoothing_cov is True
            return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks), torch.stack(Cs), Sigmas_tilde_chol, torch.stack(Sigmas_smooth)
        
        if get_sigma_tilde:
            return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks), torch.stack(Cs), Sigmas_tilde_chol
        else:
            return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks), torch.stack(Cs) # (T, b, b), (T-1, b, b), (T, b, x_dim), (T-1, b, b)
    else:
        return torch.stack(Sigmas_filt), torch.stack(Sigmas_diffused), torch.stack(Ks) # (T, b, b), (T-1, b, b), (T, b, x_dim)
    
def general_kalman_means(A, W, b, mu0, x_hat: Tensor, Ks: Tensor, Cs: Tensor, smoothing=True):
    # Ks.shape = (T, ntrials, b, x_dim)
    # Cs.shape = (T-1, ntrials, b, b)

    W_var, A_var = True, True
    if len(W.shape) == 2:
        W = W[None, None, ...] # T, ntrials, x_dim, b
        W_var = False
    if len(A.shape) == 2:
        A = A[None, None, ...] # T-1, ntrials, b, b
        A_var = False
    if len(mu0.shape) == 2:
        mu0 = mu0[None, ...] # ntrials, b

    n_mc_z, n_trials, x_dim, batch_size = x_hat.shape
    _xhat = x_hat.permute(-1, 0, 1, 2) # (batch_size, n_mc_z, ntrials, x_dim)

    # Kalman filter
    mus_filt = [] # mu_filt[t] = mu_t^t # (batch_size, n_mc_z, ntrials, b)
    mus_diffused = [] # mu_diffused[t] = mu_{t+1}^t # (batch_size-1, n_mc_z, ntrials, b)

    # Sub in the first time step
    mus_filt.append(mu0 + (Ks[0] @ ( _xhat[0][..., None] - W[0]@ mu0[..., None])).squeeze(-1)) # (n_mc_z, ntrials, b)

    for t in range(1, batch_size):
        # Find matrices relevant for this time step
        _A = A[t-1] if A_var else A[0]
        _W = W[t] if W_var else W[0]
        # populate mus_diffused[t-1] and mus_filt[t]
        mus_diffused.append((_A @ mus_filt[t-1][..., None]).squeeze(-1)) # (n_mc_z, ntrials, b)
        mus_filt.append(mus_diffused[t-1] + (Ks[t] @ (_xhat[t][..., None] - _W @ mus_diffused[t-1][..., None])).squeeze(-1)) # (n_mc_z, ntrials, b)

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
        return torch.linalg.solve_triangular(u.transpose(-1,-2), u_inv_x, upper=True)
    else:
        # Solve z U^T = x
        z_u_inv = torch.linalg.solve_triangular(u.transpose(-1,-2), x, upper=True, left=False)
        # Solve y U = z
        return torch.linalg.solve_triangular(u, z_u_inv, upper=False, left=False)