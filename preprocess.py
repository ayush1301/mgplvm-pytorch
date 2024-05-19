import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import abc
from torch.distributions import MultivariateNormal, Poisson
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch import nn
from torch.optim.lr_scheduler import StepLR
from main import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pickle
from scipy.interpolate import CubicSpline
from sklearn.metrics import r2_score
torch.set_default_dtype(torch.float64)


def main(dim, train_params, trials=5, data_len=5000, suffix='', dataset='Doherty', data_path=None, init_P = None):
    if dataset == 'Doherty':
        prefix = 'new_params/'
    elif dataset == '4g10':
        prefix = '4g10preprocess/'

    if dataset == 'Doherty':
        if data_path is None:
            data = pickle.load(open('data/Doherty_example.pickled', 'rb')) # load example data
            binsize = 25 # binsize in ms
            start = 0
            timepoints = np.arange(start, data_len) #subsample ~40 seconds of data so things will run somewhat quicker
            print(data['Y'].shape)
            fit_data = {'Y': data['Y'][..., timepoints], 'locs': data['locs'][timepoints, :], 'targets': data['targets'][timepoints, :], 'binsize': binsize}
            # fit_data = {'Y': data['Y'], 'locs': data['locs'], 'targets': data['targets'], 'binsize': binsize}
            Y = fit_data['Y'] # these are the actual recordings and is the input to our model
            targets = fit_data['targets'] # these are the target locations
            locs = fit_data['locs'] # these are the hand positions

            ts = np.arange(Y.shape[-1])*fit_data['binsize'] # measured in ms
            delay = 120
            cs = CubicSpline(ts+delay, locs) # fit cubic spline to behavior
            vels = cs(ts, 1) # velocity (first derivative)

            v = vels.T[None, ...]
        else:
            data = pickle.load(open(data_path, 'rb'))
            start = 0
            timepoints = np.arange(start, data_len)
            Y = data['Y'][..., timepoints]
            vels = data['vels'][timepoints]
            v = vels.T[None, ...]
            print('v shape', v.shape)
            print('Y shape', Y.shape)

        v = v.transpose(1,0,2).reshape(2,trials,-1).transpose(1,0,2)
        print(v.shape, 'v shape')
        final_ntrials = 1
    elif dataset == '4g10':
        data = np.load(data_path)
        v = data['hand_train']
        v = v.transpose(1,0,2)
        print('v shape', v.shape)
        final_ntrials = v.shape[0]

    p = Preprocessor(Tensor(v), dim)

    # Just trying fixed mu0 and Signa 0 half
    p.mu0.requires_grad = False
    p.Sigma0_half.requires_grad = False

    # p.W.data = torch.eye(dim).to(device)
    # p.A.data = torch.eye(dim).to(device)

    if init_P is not None:
        p.A.data = init_P.A.data
        p.W.data = init_P.W.data
        p.B.data = init_P.B.data
        p.mu0.data = init_P.mu0.data
        p.Sigma0_half.data = init_P.Sigma0_half.data
        p.R_half.data = init_P.R_half.data
    save_name = prefix + suffix
    p.train_preprocessor(p.training_params(**train_params), save_name=save_name)
    p.freeze_params()

    pickle.dump(p, open(save_name + '.pkl', 'wb'))
    
    print(p.R)

    _, _, Ks, Cs = general_kalman_covariance(p.A, p.W, p.Q, p.R, p.z_dim, p.v_dim, p.Sigma0, T=p.T, smoothing=True)
    _ , mus_smooth, _ = general_kalman_means(p.A, p.W, p.z_dim, p.mu0, Tensor(v[None, ...]).to(device), Ks, Cs=Cs, smoothing=True)
    print(mus_smooth.shape, 'mus_smooth shape')
    z_hat = mus_smooth.squeeze(1).permute(1, -1, 0).detach().cpu().numpy()
    if dataset == 'Doherty':
        z_hat = z_hat.transpose(1,0,2).reshape(dim, 1, -1).transpose(1,0,2)
    print(z_hat.shape, 'z_hat shape')

    np.save(prefix + 'z_hat_' + suffix, z_hat)


if __name__ == '__main__':
    train_params = {'batch_size': None, 'lrate': 1e-2, 'max_steps': 1001, 'step_size': 200, 'save_every': 10}
    # main(10, train_params, suffix='10ms', dataset='4g10', data_path='data_10ms.npz')
    # main(10, train_params, suffix='9k1t', dataset='Doherty', data_len=9000, trials=1)
    # p_old = pickle.load(open('new_params/_1t.pkl', 'rb'))
    # main(10, train_params, suffix='9k_new', dataset='Doherty', data_len=9000, trials=1, init_P=p_old)
    # main(20, train_params, suffix='5k_20z', dataset='Doherty', data_len=5000, trials=1)
    # main(30, train_params, suffix='5k_30z', dataset='Doherty', data_len=5000, trials=1)
    # main(30, train_params, suffix='5k_30z_new', dataset='Doherty', data_len=5000, trials=1)
    # main(5, train_params, suffix='5k_5z_new', dataset='Doherty', data_len=5000, trials=1)
    # main(15, train_params, suffix='5k_15z', dataset='Doherty', data_len=5000, trials=1)
    # main(2, train_params, suffix='5k_2z', dataset='Doherty', data_len=5000, trials=1)
    # main(20, train_params, suffix='5k_20z_new', dataset='Doherty', data_len=5000, trials=1)
    # main(2, train_params, suffix='5k_2z_better_init', dataset='Doherty', data_len=5000, trials=1)

    main(10, train_params, suffix='64ms_10z_2k', dataset='Doherty', data_len=2000, trials=1, data_path='processed_data_64.pickled')





# def main(dim, trials=5, data_len=5000, suffix='', steps=201):
#     data = pickle.load(open('data/Doherty_example.pickled', 'rb')) # load example data
#     binsize = 25 # binsize in ms
#     start = 0
#     timepoints = np.arange(start, data_len) #subsample ~40 seconds of data so things will run somewhat quicker
#     print(data['Y'].shape)
#     fit_data = {'Y': data['Y'][..., timepoints], 'locs': data['locs'][timepoints, :], 'targets': data['targets'][timepoints, :], 'binsize': binsize}
#     # fit_data = {'Y': data['Y'], 'locs': data['locs'], 'targets': data['targets'], 'binsize': binsize}
#     Y = fit_data['Y'] # these are the actual recordings and is the input to our model
#     targets = fit_data['targets'] # these are the target locations
#     locs = fit_data['locs'] # these are the hand positions

#     Y = Y[:, np.mean(Y,axis = (0, 2))/0.025 > 8, :] #subsample highly active neurons so things will run a bit quicker
#     # Y_test = Y[..., 1000:] # hold out some data for testing
#     # Y = Y[..., :1000] # use first 1000 time bins for training
#     print(Y.shape)
#     ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]

#     ts = np.arange(Y.shape[-1])*fit_data['binsize'] # measured in ms
#     cs = CubicSpline(ts, locs) # fit cubic spline to behavior
#     vels = cs(ts, 1) # velocity (first derivative)

#     v = vels.T[None, ...]
#     print(v.shape)

#     v = v.transpose(1,0,2).reshape(2,trials,-1).transpose(1,0,2)
#     print(v.shape)

#     p = Preprocessor(Tensor(v), dim)
#     train_params = {'batch_size': None, 'lrate': 1e-2, 'max_steps': steps, 'step_size': 1000}
#     p.train_preprocessor(p.training_params(**train_params))
#     p.plot_LL()
#     p.freeze_params()

#     # if suffix == '':
#     #     suffix = '_new_{}'.format(dim)
#     # pickle.dump(p, open('new_params/new_{}.pkl'.format(dim), 'wb'))
#     pickle.dump(p, open('new_params/' + suffix + '.pkl', 'wb'))
    
#     print(p.Q)
#     print(p.R)
#     print(p.Sigma0)
#     # for name, param in p.named_parameters():
#     #     print(name, param)
#     #     np.save('new_params/' + name + suffix, param.detach().cpu().numpy())

#     _, _, Ks, Cs = general_kalman_covariance(p.A, p.W, p.Q, p.R, p.z_dim, p.v_dim, p.Sigma0, T=p.T, smoothing=True)
#     _ , mus_smooth, _ = general_kalman_means(p.A, p.W, p.z_dim, p.mu0, Tensor(v[None, ...]).to(device), Ks, Cs=Cs, smoothing=True)
#     print(mus_smooth.shape)
#     z_hat = mus_smooth.squeeze(1).permute(1, -1, 0).detach().cpu().numpy()
#     print(z_hat.shape)
#     z_hat = z_hat.transpose(1,0,2).reshape(dim, 1, -1).transpose(1,0,2)
#     np.save('new_params/z_hat' + suffix, z_hat)


# if __name__ == '__main__':
#     # main(10)
#     # main(5)
#     # main(10, trials=50, data_len=5000, suffix='_50t_')
#     # main(10, trials=50, data_len=5000, suffix='50t_')
#     # main(10, trials=10, data_len=5000, suffix='_10t')
#     # main(10, trials=10, data_len=5000, suffix='_10t')
#     main(10, trials=1, data_len=5000, suffix='_1t', steps=1001)
