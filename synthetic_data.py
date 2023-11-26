import numpy as np
from matplotlib import pyplot as plt
import torch
import mgplvm as mgp
import time
from scipy.stats import poisson, norm
import pickle

# def matern_kernel(r, l, nu):
#     """
#     Matern kernel function
#     """
#     return (2**(1-nu)/gamma(nu))*((np.sqrt(2*nu)*r)/l)**nu*kv(nu, (np.sqrt(2*nu)*r)/l)


def squared_exponential_kernel(r, l):
    """
    Squared exponential kernel function
    """
    return np.exp(-(r**2) / (2 * l**2))


def exponential_kernel(r, l):
    """
    Exponential kernel function
    """
    return np.exp(-r / l)


def matern_3_2_kernel(r, l):
    """
    Matern 3/2 kernel function
    """
    return (1 + np.sqrt(3) * r / l) * np.exp(-np.sqrt(3) * r / l)


def matern_5_2_kernel(r, l):
    """
    Matern 5/2 kernel function
    """
    return (1 + np.sqrt(5) * r / l + 5 * r**2 /
            (3 * l**2)) * np.exp(-np.sqrt(5) * r / l)

def rational_quadratic_kernel(r, l, alpha):
    """
    Rational quadratic kernel function
    """
    return (1 + r**2 / (2 * alpha * l**2))**(-alpha)


class SyntheticData:

    def __init__(self,
                 dt=0.02,
                 D=3,
                 ntrials=1,
                 l=0.2,
                 start=0,
                 stop=20,
                 c_mean=0,
                 c_var=1,
                 N=100,
                 d=10,
                 noise='poisson',):
        self.dt = dt  # time step (seconds)
        self.T = int((stop - start) / dt)  # number of time steps

        self.D = D  # number of latent dimensions
        self.ntrials = ntrials  # number of trials

        if isinstance(
                l, float):  # set length scale (in seconds) for each dimension
            self.l = np.array([l] * D)
            self._same_l = True
        else:
            self.l = np.array(l)
            self._same_l = False

        self.start = start  # start time
        self.stop = stop  # stop time
        self.c_mean = c_mean  # mean of C
        self.c_var = c_var  # variance of C
        self.N = N  # number of neurons
        self.C = self.get_C()  # N x D matrix
        self.x_axis = np.linspace(self.start, self.stop, self.T)
        self.d = d  # d for CX + d

        self.Xs = []
        self.Ys = []

        self.noise = noise

    def get_X(self, kernel):  # return ntrials x D x T matrix
        if len(self.l) != self.D:
            raise ValueError("Length scale must be list of length D")

        X = np.zeros((self.ntrials, self.D, self.T))
        for i, l in enumerate(self.l):
            if kernel == 'squared_exponential':
                K = squared_exponential_kernel(
                    self.x_axis[:, None] - self.x_axis[None, :], l)
            elif kernel == 'exponential':
                K = exponential_kernel(
                    abs(self.x_axis[:, None] - self.x_axis[None, :]), l)
            elif kernel == 'matern_3_2':
                K = matern_3_2_kernel(
                    abs(self.x_axis[:, None] - self.x_axis[None, :]), l)
            elif kernel == 'matern_5_2':
                K = matern_5_2_kernel(
                    abs(self.x_axis[:, None] - self.x_axis[None, :]), l)
            elif kernel == 'rational_quadratic1':
                K = rational_quadratic_kernel(
                    self.x_axis[:, None] - self.x_axis[None, :], l, 1)
            else:
                raise ValueError("Kernel Incorrect")

            if self._same_l:  # This formulation is much faster but can only be done if all the length scales are the same
                X = np.random.multivariate_normal(mean=np.zeros(len(
                    self.x_axis)),
                                                  cov=K,
                                                  size=(self.ntrials, self.D))
                break
            else:
                X[:, i, :] = np.random.multivariate_normal(mean=np.zeros(
                    len(self.x_axis)),
                                                           cov=K,
                                                           size=(self.ntrials))

        self.Xs.append(X)
        return self.Xs[-1]

    def get_C(self):  # returns N x D matrix
        return np.random.normal(self.c_mean, self.c_var, size=(self.N, self.D))

    def get_Y(self, kernel='squared_exponential', link='softplus', s=0):
        X = self.get_X(kernel)
        F = np.matmul(self.C, X) + self.d
        if link == 'softplus':
            F = np.log(1 + np.exp(F))
        elif link == 'exp':
            F = np.exp(F)
        elif link is None:
            pass
        mu = self.dt * F

        if self.noise == 'poisson':
            self.Ys.append(np.random.poisson(mu))
        elif self.noise == 'gaussian': # Wrong
            # sig = 0.6 * np.std(F)
            # noise = np.random.normal(0, sig, size=F.shape)
            # Y = F + noise
            # self.Ys.append(Y - np.mean(Y, axis=-1, keepdims=True))
            self.Ys.append(mu + np.random.normal(0, s, size=F.shape))
        return self.Ys[-1]

    def plot_Y(self, ind=None, save_fig=None):
        ### plot the activity we just loaded ###
        for i, Y in enumerate(self.Ys):
            if ind is None or i == ind:
                plt.figure(figsize=(12, 6))
                plt.imshow(Y[0, ...],
                           cmap='Greys',
                           aspect='auto',
                           vmin=np.quantile(Y, 0.01),
                           vmax=np.quantile(Y, 0.99))
                plt.xlabel('time')
                plt.ylabel('neuron')
                plt.title('Raw activity', fontsize=25)
                plt.xticks([])
                plt.yticks([])
                if save_fig is not None:
                    plt.savefig(save_fig + '_Y_{}.png'.format(i), bbox_inches='tight')
                plt.show()

    def plot_X(self, ind=None, save_fig=None):
        ### plot the activity we just loaded ###
        for i, X in enumerate(self.Xs):
            if ind is None or i == ind:
                X = X[0, ...]  # take first trial
                for _X in X:  # Loop through dimensions
                    plt.plot(self.x_axis, _X.T)
                if save_fig is not None:
                    plt.savefig(save_fig + '_X_{}.png'.format(i), bbox_inches='tight')
                plt.show()

    def training_setup(self, ind: int, rho, n_mc, print_every, max_steps,
                       lrate, prior_ell_factor=1):
        device = mgp.utils.get_device()  # use GPU if available, otherwise CPU
        binsize = self.dt * 1000
        ts = np.arange(self.T)
        # fit_ts1 = torch.tensor(ts)[None, None, :].to(device) # Old code suitable for 1 trial
        fit_ts = torch.tensor(ts).repeat(self.ntrials, 1,
                                         1).to(device)  # shape [n_trials, 1, T]

        Y = self.Ys[ind]
        data = torch.tensor(Y).to(device)

        ### set some parameters for fitting ###
        # aa2236 need to make this configurable
        # This needs to be of type float or int!!
        ell0 = float(
            (self.l[0] * 1000) / binsize
        )  # initial timescale (in bins) for each dimension. This could be the ~timescale of the behavior of interest (otherwise a few hundred ms is a reasonable default)
        ell0 = ell0 * prior_ell_factor
        if rho is None:
            rho = np.sqrt(
                self.c_var
            )  # sets the intial scale of each latent (s_d in Jensen & Kao). rho=1 is a natural choice with Gaussian noise; less obvious with non-Gaussian noise but rho=1-5 works well empirically.
        else:
            rho = rho

        ### training will proceed for 1000 iterations (this takes ~2 minutes) ###
        t0 = time.time()

        def cb(mod, i, loss):
            """here we construct an (optional) function that helps us keep track of the training"""
            if i % print_every == 0:
                sd = np.log(mod.obs.dim_scale.detach().cpu().numpy().flatten())
                print('iter:', i, 'time:',
                      str(round(time.time() - t0)) + 's', 'log scales:',
                      np.round(sd[np.argsort(-sd)], 1))
                # print(loss.item(), mod.calc_LL(data = data, n_mc = n_mc).item())

        return Y, fit_ts, rho, ell0, device, data, cb

    def train(self,
              ind: int,
              nu: bool = False,
              rho=None,
              lrate=7.5e-2,
              max_steps=1001,
              n_mc=5,
              print_every=100,
              d_fit=10,
              prior_fourier_func=None,
            #   inv_link=mgp.utils.softplus,
              likelihood_kwargs=None,
              prior_ell_factor=1, save_fig=None):
        Y, fit_ts, rho, ell0, device, data, cb = self.training_setup(
            ind, rho, n_mc, print_every, max_steps, lrate, prior_ell_factor)

        ### construct the actual model ###
        ntrials, n, T = Y.shape  # Y should have shape: [number of trials (here 1) x neurons x time points]
        # lik = mgp.likelihoods.NegativeBinomial(n, Y=Y) # we use a negative binomial noise model in this example (recommended for ephys data)
        if self.noise == 'poisson':
            lik = mgp.likelihoods.Poisson(n, **likelihood_kwargs)
        elif self.noise == 'gaussian':
            lik = mgp.likelihoods.Gaussian(n, Y=Y, d=d_fit)

        manif = mgp.manifolds.Euclid(
            T, d_fit
        )  # our latent variables live in a Euclidean space for bGPFA (see Jensen et al. 2020 for alternatives)
        var_dist = mgp.rdist.GP_circ(
            manif,
            T,
            ntrials,
            fit_ts,
            _scale=1,
            ell=ell0,
            prior_fourier_func=prior_fourier_func
        )  # circulant variational GP posterior (c.f. Jensen & Kao et al. 2021)
        lprior = mgp.lpriors.Null(
            manif
        )  # here the prior is defined implicitly in our variational distribution, but if we wanted to fit e.g. Factor analysis this would be a Gaussian prior
        mod = mgp.models.Lvgplvm(n,
                                 T,
                                 d_fit,
                                 ntrials,
                                 var_dist,
                                 lprior,
                                 lik,
                                 Y=Y,
                                 learn_scale=False,
                                 ard=True,
                                 rel_scale=rho).to(
                                     device)  #create bGPFA model with ARD

        # max_steps = 1001 # number of training iterations
        # n_mc = 5 # number of monte carlo samples per iteration
        # print_every = 100 # how often we print training progress

        # helper function to specify training parameters
        train_ps = mgp.crossval.training_params(max_steps=max_steps,
                                                n_mc=n_mc,
                                                lrate=lrate,
                                                callback=cb,
                                                burnin=50)
        print('fitting', n, 'neurons and', T, 'time bins for', max_steps,
              'iterations')
        mod_train = mgp.crossval.train_model(mod, data, train_ps)

        if nu is None:
            self._plot_dims(mod, True, save_fig=save_fig)
            self._plot_dims(mod, False, save_fig=save_fig)
        else:
            self._plot_dims(mod, nu, save_fig=save_fig)

        return mod
    
    def _plot_dims(self, mod, nu, save_fig=None):
        ### we start by plotting 'informative' and 'discarded' dimensions ###
        print('plotting informative and discarded dimensions')
        dim_scales = mod.obs.dim_scale.detach().cpu().numpy().flatten(
        )  #prior scales (s_d)
        dim_scales = np.log(dim_scales)  #take the log of the prior scales
        plt.figure()
        if nu:
            nus = np.sqrt(
                np.mean(mod.lat_dist.nu.detach().cpu().numpy()**2,
                        axis=(0, -1)))  #magnitude of the variational means
            plt.scatter(
                dim_scales, nus, c='k', marker='x',
                s=80)  #top right corner are informative, lower left discarded
            plt.ylabel(r'$||\nu||_2$', labelpad=5)
        else:
            # nu is shape [n_trials, n_dims, n_times]
            # lat_mu is shape [n_trials, n_times, n_dims]
            lat_mu = mod.lat_dist.lat_mu.detach().cpu().numpy()
            mus = np.sqrt(np.mean(lat_mu**2, axis=(0, 1)))
            plt.scatter(dim_scales, mus, c='k', marker='x', s=80)
            plt.ylabel(r'$||\mu||_2$', labelpad=5)

        plt.xlabel(r'$\log \, s_d$')
        if save_fig is not None:
            if nu:
                suffix = '_nu'
            else:
                suffix = '_mu'
            plt.savefig(save_fig + '{}.png'.format(suffix), bbox_inches='tight')
        plt.show()

    def cross_validate(self,
                       ind: int,
                       nu: bool = False,
                       rho=None,
                       lrate=7.5e-2,
                       max_steps=1001,
                       n_mc=5,
                       print_every=100,
                       burnin=50,
                       d_fit=10,
                       nt_train=None,
                       nn_train=None,
                       prior_fourier_func=None,
                       likelihood='Poisson',
                       likelihood_kwargs=None,
                       prior_ell_factor=1,
                       save_mod=None,
                       save_fig=None):
        Y, fit_ts, rho, ell0, device, _, cb = self.training_setup(
            ind, rho, n_mc, print_every, max_steps, lrate, prior_ell_factor)

        train_ps = mgp.crossval.training_params(max_steps=max_steps,
                                                n_mc=n_mc,
                                                burnin=burnin,
                                                lrate=lrate,
                                                callback=cb)
        _lik = 'Poisson' if self.noise == 'poisson' else 'Gaussian'
        mod, split, trained = mgp.crossval.train_cv_bgpfa(
            Y=Y,
            device=device,
            train_ps=train_ps,
            fit_ts=fit_ts,
            d_fit=d_fit,
            ell=ell0,
            likelihood=_lik,
            nt_train=nt_train,
            nn_train=nn_train,
            test=False,
            rel_scale=rho,
            prior_fourier_func=prior_fourier_func,
            likelihood_kwargs=likelihood_kwargs)
        ### need to compute predictive likelihood ###
        if save_mod is not None:
            pickle.dump(mod, open(save_mod + 'model.pickled', 'wb'))
        Ycv, T1cv, N1cv = split['Y'], split['T1'], split[
            'N1']  # all data, train data and train neurons
        m = self.T
        n = self.N
        T2cv, N2cv = mgp.crossval.crossval.not_in(
            np.arange(m), T1cv), mgp.crossval.crossval.not_in(
                np.arange(n),
                N1cv)  # these are our testing timepoints and neurons

        Ytest = Ycv[:, N2cv, :][..., T2cv]  #test data (shape ntrial x N2 x T2)

        latents = mod.lat_dist.lat_mu.detach().cpu(
        )[:, T2cv, ...]  # extract test latent means (shape ntrial, T2, d)
        # construct input for prediction of neural activity
        query = latents.transpose(-1, -2).to(device)  #(ntrial, d, m)

        # predicted 'F' # aa2236 this is without transfer function so it can be negative
        mu = mod.svgp.predict(query[None, ...], full_cov=False)[0]
        # mean of the implied distribution after transfer function (for test neurons)
        Ypred = mod.svgp.likelihood.dist_mean(
            mu)[0].detach().cpu().numpy()[:, N2cv, :]
        
        # ## aa2236 from bgpfa.ipynb
        # Ypreds = []
        # for i in range(10): # loop over mc samples to avoid memory issues
        #     Ypred2 = mod.svgp.sample(query, n_mc=100, noise=False) # OG n_mc = 100
        #     Ypred2 = Ypred2.detach().mean(0).cpu().numpy()  # (ntrial x n x T)
        #     Ypreds.append(Ypred2)
        # Ypred2 = np.mean(np.array(Ypreds), axis = (0,1)).T # T x n
        # ## aa2236


        # compute log probability of actual test spike counts given test mean predictions
        if self.noise == 'poisson':
            LL = poisson.logpmf(Ytest, Ypred, loc=0)
            LL = np.mean(LL)  # take avg
        elif self.noise == 'gaussian':
            LL = np.mean(np.log(
                norm.pdf(Ytest, loc=Ypred, scale=np.std(Ypred))))

        # also compute MSEs
        print(Ypred.shape, Ytest.shape, 'Ypred, Ytest')
        MSE_vals = np.mean((Ypred - Ytest)**2, axis=(0, -1))
        MSE = np.mean(MSE_vals)  #standard MSE

        # aa2236 use self.obs or self.svgp?
        # svgp_lik = mod.obs.variational_expectation()

        if nu is None:
            self._plot_dims(mod, True, save_fig=save_fig)
            self._plot_dims(mod, False, save_fig=save_fig)
        else:
            self._plot_dims(mod, nu, save_fig=save_fig)
        
        lat_traj, _ = self.get_lat_traj(ind, mod)

        return LL, MSE, trained[-1], lat_traj  # LL, MSE, final loss
    
    def get_lat_traj(self, ind, mod=None):
        # assuming just 1 trial
        xs = self.Xs[ind][0].T # true latents
        # print(np.shape(xs))
        xs = xs - np.mean(xs, axis = 0, keepdims = True) # mean subtract

        if mod is not None:
            #### plot latents ####
            lim = 100
            # plt.figure()
            # plt.plot(xs[:lim, 0], xs[:lim, 1], 'k-', label='true') # plot true latents
            lats = mod.lat_dist.lat_mu.detach().cpu().numpy()[0, ...] # extract model latents
            dim_scales = mod.obs.dim_scale.detach().cpu().numpy().flatten() # latent scales
            inds = np.argsort(-dim_scales)[:2] # pick the two most informative dimensions
            lats = lats[..., inds] # extract corresponding latents
            lats = lats - np.mean(lats, axis = 0, keepdims = True) # mean subtract

            #fit xs = mus @ T --> T = (mus' * mus)^(-1) * mus' * xs
            T = np.linalg.inv(lats.T @ lats) @ lats.T @ xs # linear regression
            lats = lats @ T  #predicted values

            return lats, xs
        else:
            return xs
