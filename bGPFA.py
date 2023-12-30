import torch
import mgplvm as mgp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
np.random.seed(0)
torch.manual_seed(0)
device = mgp.utils.get_device() # use GPU if available, otherwise CPU
from scipy.stats import poisson

def main(kernels, seeds, data_len, batch_size=None, n_mc=1, pre_prefix=''):
    data = pickle.load(open('data/Doherty_example.pickled', 'rb')) # load example data
    binsize = 25 # binsize in ms
    timepoints = np.arange(0, data_len) #subsample ~40 seconds of data so things will run somewhat quicker
    print(data['Y'].shape)
    fit_data = {'Y': data['Y'][..., timepoints], 'locs': data['locs'][timepoints, :], 'targets': data['targets'][timepoints, :], 'binsize': binsize}
    # fit_data = {'Y': data['Y'], 'locs': data['locs'], 'targets': data['targets'], 'binsize': binsize}
    Y = fit_data['Y'] # these are the actual recordings and is the input to our model
    targets = fit_data['targets'] # these are the target locations
    locs = fit_data['locs'] # these are the hand positions
    # print(Y.shape)
    Y = Y[:, np.mean(Y,axis = (0, 2))/0.025 > 2, :] #subsample highly active neurons so things will run a bit quicker
    # print(Y.shape)
    ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]
    data = torch.tensor(Y).to(device) # put the data on our GPU/CPU
    ts = np.arange(Y.shape[-1]) #much easier to work in units of time bins here
    fit_ts = torch.tensor(ts)[None, None, :].to(device) # put our time points on GPU/CPU

    # finally let's just identify bins where the target changes
    deltas = np.concatenate([np.zeros(1), np.sum(np.abs(targets[1:, :] - targets[:-1, :]), axis = 1)])
    switches = np.where(deltas > 1e-5)[0] # change occurs during time bin s
    dswitches = np.concatenate([np.ones(1)*10, switches[1:] - switches[:-1]]) # when the target changes during a bin there will be two discontinuities
    inds = np.zeros(len(switches)).astype(bool)
    inds[dswitches > 1.5] = 1 # index of the bin where the target changes or the first bin with a new target
    switches = switches[inds]

    # print(np.mean(Y, axis = (0, 2)))

    ### set some parameters for fitting ###
    ell0 = 200/binsize # initial timescale (in bins) for each dimension. This could be the ~timescale of the behavior of interest (otherwise a few hundred ms is a reasonable default)
    rho = 2 # sets the intial scale of each latent (s_d in Jensen & Kao). rho=1 is a natural choice with Gaussian noise; less obvious with non-Gaussian noise but rho=1-5 works well empirically.
    max_steps = 1001 # number of training iterations
    # n_mc = 5 # number of monte carlo samples per iteration
    print_every = 100 # how often we print training progress
    d_fit = 20 # lets fit up to 1


    ### construct the actual model ###
    ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]

    ts = np.arange(Y.shape[-1])*fit_data['binsize'] # measured in ms
    cs = CubicSpline(ts, locs) # fit cubic spline to behavior
    vels = cs(ts, 1) # velocity (first derivative)

    # aa2236 for looping
    seeds = np.arange(seeds) # we'll fit 5 models with different random initializations

    R2s = np.zeros((len(kernels), len(seeds)))
    for i_k, kernel in enumerate(kernels):
        for seed in seeds:
            torch.cuda.empty_cache() # clear GPU memory

            name_prefix = pre_prefix + 'kernel_{}_seed_{}'.format(i_k, seed)
            print('kernel: ', i_k, 'seed: ', seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            lik = mgp.likelihoods.NegativeBinomial(n, Y=Y) # we use a negative binomial noise model in this example (recommended for ephys data)
            # lik = mgp.likelihoods.Poisson(n, d=15*torch.ones(n,), inv_link=mgp.utils.softplus) # we use a negative binomial noise model in this example (recommended for ephys data)
            manif = mgp.manifolds.Euclid(T, d_fit) # our latent variables live in a Euclidean space for bGPFA (see Jensen et al. 2020 for alternatives)
            var_dist = mgp.rdist.GP_circ(manif, T, ntrials, fit_ts, _scale=1, ell = ell0, prior_fourier_func=kernel) # circulant variational GP posterior (c.f. Jensen & Kao et al. 2021)
            lprior = mgp.lpriors.Null(manif) # here the prior is defined implicitly in our variational distribution, but if we wanted to fit e.g. Factor analysis this would be a Gaussian prior
            mod = mgp.models.Lvgplvm(n, T, d_fit, ntrials, var_dist, lprior, lik, Y = Y, learn_scale = False, ard = True, rel_scale = rho).to(device) #create bGPFA model with ARD

            t0 = time.time()
            def cb(mod, i, loss):
                """here we construct an (optional) function that helps us keep track of the training"""
                if i % print_every == 0:
                    sd = np.log(mod.obs.dim_scale.detach().cpu().numpy().flatten())
                    print('iter:', i, 'time:', str(round(time.time()-t0))+'s', 'log scales:', np.round(sd[np.argsort(-sd)], 1))
                    # print(loss, mod.calc_LL(data = data, n_mc = n_mc))
            # helper function to specify training parameters
            train_ps = mgp.crossval.training_params(max_steps = max_steps, n_mc = n_mc, lrate = 5e-2, callback = cb, print_every=1, batch_size = batch_size)
            print('fitting', n, 'neurons and', T, 'time bins for', max_steps, 'iterations')
            torch.cuda.empty_cache() # clear GPU memory
            mod_train = mgp.crossval.train_model(mod, data, train_ps)

            ### we start by plotting 'informative' and 'discarded' dimensions ###
            print('plotting informative and discarded dimensions')
            dim_scales = mod.obs.dim_scale.detach().cpu().numpy().flatten() #prior scales (s_d)
            dim_scales = np.log(dim_scales) #take the log of the prior scales
            nus = np.sqrt(np.mean(mod.lat_dist.nu.detach().cpu().numpy()**2, axis = (0, -1))) #magnitude of the variational mean
            plt.figure()
            plt.scatter(dim_scales, nus, c = 'k', marker = 'x', s = 80) #top right corner are informative, lower left discarded
            plt.xlabel(r'$\log \, s_d$')
            plt.ylabel('latent mean scale', labelpad = 5)
            plt.savefig('imgs/' + name_prefix + 'dims.png', dpi = 300, bbox_inches = 'tight')
            plt.close()

                        ### plot the inferred latent trajectories
            print('plotting latent trajectories')
            X = mod.lat_dist.lat_mu.detach().cpu().numpy()[0, ...] # extract inferred latents ('mu' has shape (ntrials x T x d_fit))
            X = X[..., np.argsort(-dim_scales)] # only consider the two most informative dimensions (c.f. Jensen & Kao)
            tplot = np.arange(300, 400) # let's only plot a shorter period (here 2.s) so it doesn't get too cluttered

            # fit FA for comparison
            fa = FactorAnalysis(2)
            Xfa = fa.fit_transform(np.sqrt(Y[0, ...].T)) # sqrt the counts for variance stabilization (c.f. Yu et al. 2009)

            i1, i2 = 2, 3 # which dimensions to plot
            fig, axs = plt.subplots(1, 2, figsize = (10, 5))
            axs[0].scatter(X[tplot, i1], X[tplot, i2], c = tplot, cmap = 'coolwarm', s = 80) # plot bGPFA latents
            axs[1].scatter(Xfa[tplot, 0], Xfa[tplot, 1], c = tplot, cmap = 'coolwarm', s = 80) # plot FA latents
            for ax in axs:
                ax.set_xlabel('latent dim 1')
                ax.set_ylabel('latent dim 2')
                ax.set_xticks([])
                ax.set_yticks([])
            axs[0].set_title('Bayesian GPFA')
            axs[1].set_title('factor analysis')
            plt.savefig('imgs/' + name_prefix + 'latent_traj.png', dpi = 300, bbox_inches = 'tight')
            plt.close()

            # let's also print the learned timescales (sorted by the prior scales s_d)
            taus = mod.lat_dist.ell.detach().cpu().numpy().flatten()[np.argsort(-dim_scales)]*binsize
            print('learned timescales (ms):', np.round(taus).astype(int))

            ### finally let's do a simple decoding analysis ###
            torch.cuda.empty_cache() # clear GPU memory
            print('running decoding analysis')
            Ypreds = [] # decode from the inferred firing rates (this is a non-linear decoder from latents)
            query = mod.lat_dist.lat_mu.detach().transpose(-1, -2).to(device)  # (ntrial, d_fit, T)
            for i in range(100): # loop over mc samples to avoid memory issues
                Ypred = mod.svgp.sample(query, n_mc=10, noise=False) # OG n_mc = 100
                Ypred = Ypred.detach().mean(0).cpu().numpy()  # (ntrial x n x T)
                Ypreds.append(Ypred)
            Ypred = np.mean(np.array(Ypreds), axis = (0,1)).T # T x n

            delays = np.linspace(-150, 250, 50) # consider different behavioral delays
            performance = np.zeros((len(delays), 2)) # model performance
            for idelay, delay in enumerate(delays):
                vels = cs(ts+delay, 1) # velocity at time+delay
                for itest, Ytest in enumerate([Ypred]): # bGPFA
                    # regs = [Ridge(alpha=1e-3).fit(Ytest[::2, :], vels[::2, i]) for i in range(2)] # fit x and y vel on half the data
                    # scores = [regs[i].score(Ytest[1::2, :], vels[1::2, i]) for i in range(2)] # score x and y vel on the other half
                    regs = [Ridge(alpha=1e-3).fit(Ytest[2400:, :], vels[2400:, i]) for i in range(2)] # fit x and y vel on half the data
                    scores = [regs[i].score(Ytest[:2400, :], vels[:2400, i]) for i in range(2)] # score x and y vel on the other half
                    # regs = [Ridge(alpha=1e-3).fit(Ytest, vels[:, i]) for i in range(2)]
                    # scores = [regs[i].score(Ytest, vels[:, i]) for i in range(2)]
                    performance[idelay, itest] = np.mean(scores) # save performance
            print('plotting decoding')
            plt.figure()
            plt.plot(delays, performance[:, 0], 'k-')
            print('R^2 for kernel {}, seed {}'.format(i_k, seed), max(performance[:, 0]))
            R2s[i_k, seed] = max(performance[:, 0])
            plt.axvline(delays[np.argmax(performance[:, 0])], color = 'b', ls = '--')
            plt.xlim(delays[0], delays[-1])
            plt.xlabel('delay (ms)')
            plt.ylabel('kinematic decoding')
            plt.savefig('imgs/' + name_prefix + 'decoding_delay.png', dpi = 300, bbox_inches = 'tight')
            plt.close()

            start = 1200 #0//binsize
            end = 1600 # 50000//binsize
            best_delay = delays[np.argmax(performance[:, 0])]
            vels = cs(ts+best_delay, 1)
            regs = [Ridge(alpha=1e-3).fit(Ypred, vels[:, i]) for i in range(2)]
            preds = regs[0].predict(Ypred[start:end, :])
            # Y_new = Y[0].T
            # regs = [Ridge(alpha=1e-3).fit(Y_new, vels[:, i]) for i in range(2)]
            # preds = regs[0].predict(Y_new[start:end, :])
            x_axis = np.arange(start, end)*binsize/1000
            plt.plot(x_axis, vels[start:end, 0], label='original v_x')
            # plt.plot(x_axis, vels[start:end, 1], label='v_y')
            plt.plot(x_axis, preds, label='bGPFA predicted v_x')
            plt.legend()
            plt.xlabel('time (s)')
            plt.ylabel('velocity (a.u.)')
            plt.savefig('imgs/' + name_prefix + 'vels.png', dpi = 300, bbox_inches = 'tight')
            plt.close()

            start = 1200 #0//binsize
            end = 1600 # 50000//binsize
            delta_t = 0.025
            x_axis = np.arange(start, end)*binsize/1000
            plt.plot(x_axis, Ypred[start:end, 0]/delta_t, label='1st neuron')
            plt.plot(x_axis, Ypred[start:end, 12]/delta_t, label='12th neuron')
            # plt.plot(Ypred[start:end, 110]/delta_t, label='12th neuron')
            plt.legend()
            plt.xlabel('time (s)')
            plt.ylabel('firing rate (Hz)')
            plt.ylim(0, 50)
            plt.savefig('imgs/' + name_prefix + 'firing_rates.png', dpi = 300, bbox_inches = 'tight')
            plt.close()

            # Picke and save the model
            pickle.dump(mod, open('models/' + name_prefix + 'model.pickled', 'wb'))
        
        # print('Kernel {} done'.format(i_k))
        # print('R2s: ', R2s[i_k, :])
        # print('Mean R2: ', np.mean(R2s[i_k, :]))
        # print('Std R2: ', np.std(R2s[i_k, :]))
        # print('Max R2: ', np.max(R2s[i_k, :]))
        # print('Min R2: ', np.min(R2s[i_k, :]))
        with open('results.txt', 'a') as f:
            f.write('Kernel {} done'.format(i_k))
            f.write('\n')
            f.write('R2s: {}'.format(R2s[i_k, :]))
            f.write('\n')
            f.write('Mean R2: {}'.format(np.mean(R2s[i_k, :])))
            f.write('\n')
            f.write('Std R2: {}'.format(np.std(R2s[i_k, :])))
            f.write('\n')
            f.write('Max R2: {}'.format(np.max(R2s[i_k, :])))
            f.write('\n')
            f.write('Min R2: {}'.format(np.min(R2s[i_k, :])))
            f.write('\n')

    for i_k, kernel in enumerate(kernels):
        # print('Kernel {}'.format(i_k), 'R2 = {} +- {}'.format(np.mean(R2s[i_k, :]), np.std(R2s[i_k, :])))
        with open('results.txt', 'a') as f:
            f.write('Kernel {}'.format(i_k) + 'R2 = {} +- {}'.format(np.mean(R2s[i_k, :]), np.std(R2s[i_k, :])))
            f.write('\n')

if __name__ == '__main__':
    # kernels = [mgp.rdist.prior_kernels.k_1_2_squared_exponential, mgp.rdist.prior_kernels.k_1_2_rational_quadratic_1, mgp.rdist.prior_kernels.k_1_2_matern_3_2, mgp.rdist.prior_kernels.k_1_2_exponential]
    # main(kernels, 5, 24000, 5000, 5)

    # kernels = [mgp.rdist.prior_kernels.k_1_2_squared_exponential]
    # main(kernels, 1, 50000, 1000, 20, '50k_')

    kernels = [mgp.rdist.prior_kernels.k_1_2_rational_quadratic_1]
    main(kernels, 1, 50000, 2000, 10, '50k_r1_')
