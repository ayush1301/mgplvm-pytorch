import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, kv
from numpy import matlib

import torch
import mgplvm as mgp
import pickle
import time
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
np.random.seed(0)
torch.manual_seed(0)
device = mgp.utils.get_device() # use GPU if available, otherwise CPU
print(device)

from scipy.stats import poisson
from synthetic_data import *

import sys
import io

def main(c_var=10, noise_std=0.5, prefix='gaussian', stop=40, bias_d=15, rho=1, max_steps=2501, poisson=False):
    if not poisson:
        s = SyntheticData(c_var=c_var, D=2, stop=stop, ntrials=1, N=200, d=bias_d, dt=0.02, noise='gaussian')

        Y = s.get_Y(kernel='squared_exponential', s=noise_std, link=None)
        print(np.shape(Y))

        # Y2 = s.get_Y(kernel='squared_exponential', noise='gaussian', s=0.1)
        # print(np.shape(Y))

        Y_e = s.get_Y(kernel='exponential', s=noise_std, link=None)
        print(np.shape(Y_e))

        Y_matern_3_2 = s.get_Y(kernel='matern_3_2', s=noise_std, link=None)
        print(np.shape(Y_matern_3_2))

        # Y_matern_5_2 = s.get_Y(kernel='matern_5_2', s=.5)
        # print(np.shape(Y_matern_5_2))

        Y_rational_quadratic_1 = s.get_Y(kernel='rational_quadratic1', s=noise_std, link=None)
        print(np.shape(Y_rational_quadratic_1))

    if poisson:
        s = SyntheticData(c_var=c_var, D=2, stop=stop, ntrials=1, N=200, d=bias_d, dt=0.02, noise='poisson')

        Y = s.get_Y(kernel='squared_exponential')
        print(np.shape(Y))

        # Y2 = s.get_Y(kernel='squared_exponential', noise='gaussian', s=0.1)
        # print(np.shape(Y))

        Y_e = s.get_Y(kernel='exponential')
        print(np.shape(Y_e))

        Y_matern_3_2 = s.get_Y(kernel='matern_3_2')
        print(np.shape(Y_matern_3_2))

        # Y_matern_5_2 = s.get_Y(kernel='matern_5_2', s=.5)
        # print(np.shape(Y_matern_5_2))

        Y_rational_quadratic_1 = s.get_Y(kernel='rational_quadratic1')
        print(np.shape(Y_rational_quadratic_1))

    s.plot_X(save_fig='imgs/mse/' + prefix)
    s.plot_Y(save_fig='imgs/mse/' + prefix)


    plt.plot(s.Ys[0][0,0], label='squared_exponential')
    plt.plot(s.Ys[1][0,0], label='exponential')
    plt.legend()
    plt.savefig('imgs/mse/' + prefix + '_Y_graph.png', bbox_inches='tight')
    plt.close()

    kernels = [mgp.rdist.prior_kernels.k_1_2_squared_exponential, mgp.rdist.prior_kernels.k_1_2_exponential, mgp.rdist.prior_kernels.k_1_2_matern_3_2, mgp.rdist.prior_kernels.k_1_2_rational_quadratic_1]
    names = ['SE', 'OU', 'M_3_2', 'R1']
    
    seeds = 5
    for data_ind in range(4):
        file_path = "{}_data.txt".format(names[data_ind])
        latent_traj_data = dict()
        for kernel in range(4):
            latent_traj_data[kernel] = dict()
            MSEs = []
            losses = []
            LLs = []
            for i in range(seeds): # trials
                # Capture the function's output
                output_buffer = io.StringIO()
                sys.stdout = output_buffer

                np.random.seed(i)
                torch.manual_seed(i)
                mod_str = 'models/mse/' + prefix + '_{}_kernel_{}_seed_{}'.format(data_ind, kernel, i)
                im_str = 'imgs/mse/' + prefix + '_{}_kernel_{}_seed_{}'.format(data_ind, kernel, i)
                LL, MSE, final_loss, lat_traj = s.cross_validate(data_ind, nu=None, rho=rho, prior_ell_factor=0.8, lrate=7.5e-2, max_steps=max_steps, save_mod=mod_str, save_fig=im_str, prior_fourier_func=kernels[kernel], likelihood_kwargs={'inv_link': mgp.utils.softplus, 'd': torch.ones(s.N,)*s.d, 'fixed_c': True, 'fixed_d': False})
                MSEs.append(MSE)
                losses.append(final_loss)
                LLs.append(LL)
                print(LL, MSE)
                print('seed {} done'.format(i))
                print('', end='\n\n')

                # Reset the standard output
                sys.stdout = sys.__stdout__
                # Get the captured output as a string
                captured_output = output_buffer.getvalue()
                # Open the file in write mode and write the captured output
                with open(file_path, 'a') as file:
                    file.write(captured_output)

                latent_traj_data[kernel][i] = lat_traj
                
            with open(file_path, 'a') as file:
                file.write('\n {} kernel done \n \n \n'.format(names[kernel]))

            with open('results.txt', 'a') as file:
                file.write('{} Data {} kernel - MSE: {} +/- {} - Loss: {} +/- {}, LL = {} +/- {} \n'.format(names[data_ind], names[kernel], np.mean(MSEs), np.std(MSEs), np.mean(losses), np.std(losses), np.mean(LLs), np.std(LLs)))
                file.write('MSEs' + str(MSEs) + '\n')
                file.write('losses' + str(losses) + '\n')
                file.write('LLs' + str(LLs) + '\n')
                file.write('\n')

        lim = 100
        true_traj = s.get_lat_traj(data_ind)
        # plt.plot(true_traj[:lim,0], true_traj[:lim,1], 'k-', label='true')

        for seed in range(seeds):
            plt.plot(true_traj[:lim,0], true_traj[:lim,1], 'k-', label='true')
            for kernel, seed_dat in latent_traj_data.items():
                lat_traj = seed_dat[seed]
                plt.plot(lat_traj[:lim,0], lat_traj[:lim,1], '-', label='{}'.format(names[kernel]))
            plt.legend()
            plt.savefig('imgs/mse/' + prefix + '_{}_seed_{}_latent_traj.png'.format(names[data_ind], seed), bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    # main()
    # main(c_var=1, noise_std=0.01, prefix='no_ard') # also Gaussian by mistake
    # main(c_var=10, noise_std=1, prefix='new_', stop=10, bias_d=0, rho=0.1, max_steps=1001) # also Gaussian by mistake
    # main(c_var=10, noise_std=0.5, prefix='new_', stop=40, bias_d=0, rho=0.1, max_steps=2001)
    # main(c_var=10, noise_std=0.5, prefix='poisson_', stop=10, bias_d=15, rho=0.1, max_steps=2001) # Poisson
    # main(c_var=10, noise_std=0.05, prefix='gsmall_', stop=10, bias_d=0, rho=0.1, max_steps=2001, poisson=False) # Gaussian small variance
    main(c_var=10, noise_std=1, prefix='gbig_', stop=10, bias_d=0, rho=0.1, max_steps=2001, poisson=False) # Gaussian big variance
