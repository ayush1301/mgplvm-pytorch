# load some libraries
import numpy as np
import matplotlib.pyplot as plt
import mgplvm as mgp
import torch
import time
import pickle
from sklearn.decomposition import FactorAnalysis
from scipy.stats import binned_statistic
from scipy.stats import poisson, norm
from .. import basedir, detach, not_in
from .load_joey import load_data

torch.set_default_dtype(torch.float64)
device = mgp.utils.get_device("cuda")  # get_device("cpu")
np.random.seed(11101401)
torch.manual_seed(11101401)

##### data parameters ######
name = 'indy_20160426_01'
region = 'M1'
region = 'both'

### model parameters (we hardcode the model selection instead of running sequentially or using the command line)###
model = 'gpfa'

model = 'bgpfa_ard'

model = 'fa'

model = 'bgpfa'

model = 'gplvm'

binsize = 25
max_steps = 3001 if model == 'bgpfa_ard' else 2501
nmax = max_steps
cv = True  # want to crossvalidate
#cv = False

_scale = 1.0
rel_scale = 3

# extract a subset of data so we can run our analyses faster
tmin, tmax = 6000, 7000

#### load data ####
Y = load_data(name=name,
              region=region,
              binsize=binsize,
              subsamp=False,
              behavior=False,
              shift=False,
              thresh=2,
              shiftsize=0)

Y = Y[..., tmin:tmax].astype(float)
data = torch.tensor(Y).to(device)

n_samples, n, T = Y.shape

nfold = 10  # we run nfold-crossvalidation across neurons
Ns = np.arange(n)
np.random.shuffle(Ns)
folds = [Ns[i::nfold] for i in range(nfold)]  # generate folds

# split into train/test timepoints
T1 = np.arange(T)[:int(round(T / 2))]
T2 = not_in(np.arange(T), T1)

# all timepoints
ts = np.arange(T)
fit_ts = torch.tensor(ts)[None, None, :].to(device)

nreps = 5  # number of repetitions (random seeds)

# number of dimensionalities to fir
ds_fit = [1, 2, 4, 6, 8, 10, 12]

# initialize arrays for storing results
LLs = np.zeros((nreps, len(ds_fit), nfold))
norm_MSEs = np.zeros((nreps, len(ds_fit), nfold))
MSEs = np.zeros((nreps, len(ds_fit), nfold))
LL_trains = np.zeros((nreps, len(ds_fit), nfold))
LL_marg = np.zeros((nreps, len(ds_fit), nfold))

t0 = time.time()


def cb_ard(mod, i, loss):
    """function for tracking progress during training"""
    if i % 400 == 0:
        print('')
        global t0
        ls, ss = detach(mod.obs.dim_scale).flatten()**(-1), np.mean(
            detach(mod.lat_dist.scale), axis=(0, -1))
        lambdas = ls**(-2)  #compute participation ratio
        dim = np.sum(lambdas)**2 / np.sum(lambdas**2)
        ms = np.sqrt(np.mean(detach(mod.lat_dist.nu)**2, axis=(0, -1)))
        args = np.argsort(ls)
        print(region, np.round(ls[args], 2), '', np.round(ss[args], 2), '',
              np.round(ms[args], 2), np.round(time.time() - t0))
        t0 = time.time()
    return False


# filename to save to
savename_cv = 'data-analysis/bgpfa_primate/rebut/comp4_model_' + model

print('\n\n')

# iterate across repetitions and folds
for nrep in range(nreps):
    for ifold in range(nfold):
        N2 = folds[ifold]  #test neurons
        N1 = np.concatenate([folds[i] for i in range(nfold) if i != ifold
                            ])  # train neurons

        Ytrain = Y[..., T1]  # extract training time points
        print('\n', Y.shape, np.var(Y[0, N2, :][:, T2]), len(N1), len(T1),
              len(N2), len(T2))

        print('nrep:', nrep, ' ifold:', ifold, len(N1), len(N2))

        iter_ = iter([ds_fit[-1]]) if model == 'bgpfa_ard' else iter(
            ds_fit)  # if using bGPFA with ARD, only fit largest dimensionality
        for dnum, d_fit in enumerate(iter_):  # for each dimensinoality

            manif = mgp.manifolds.Euclid(T, d_fit)  # Euclidean latent space
            lik = mgp.likelihoods.Poisson(n)  # Poisson model

            if model == 'fa':  #Poisson FA
                # start by fitting Gaussian FA
                fa = FactorAnalysis(n_components=d_fit)
                fa.fit_transform(Ytrain[0, ...])
                mutrain = fa.components_.T  #initialize latents from Gaussian FA (ntrial x mtrain x d)
                mu = np.concatenate([
                    mutrain,
                    np.zeros((Y.shape[-1] - Ytrain.shape[-1], d_fit))
                ])[None, ...]
                lprior = mgp.lpriors.Uniform(manif)  # uniform prior
                # Gaussian variational dist
                lat_dist = mgp.rdist.ReLie(manif,
                                           T,
                                           n_samples,
                                           sigma=0.2,
                                           diagonal=True,
                                           initialization='fa',
                                           Y=Y,
                                           mu=mu)
                # instantiate model
                mod = mgp.models.Lvgplvm(n,
                                         T,
                                         d_fit,
                                         n_samples,
                                         lat_dist,
                                         lprior,
                                         lik,
                                         Y=Ytrain,
                                         Bayesian=False,
                                         rel_scale=rel_scale * 0.1).to(device)

            elif model in ['gpfa', 'bgpfa', 'bgpfa_ard',
                           'gplvm']:  # some version of GPFA
                Bayesian = True  # most of them Bayesian (in the factor matrix)
                if model == 'gpfa':  # GPFA is not
                    Bayesian = False

            cb = cb_ard if model == 'bgpfa_ard' else None

            n_mc = 10  # number of monte carlo samples to use
            # construct parameter dict
            train_ps = mgp.crossval.training_params(max_steps=nmax,
                                                    n_mc=n_mc,
                                                    burnin=50,
                                                    lrate=5e-2,
                                                    callback=cb)

            # note that we don't use the built-in testing quantification, but only the training and prediction steps
            if model == 'fa':
                # perform crossvalidated training using bGPFA library
                mod, split = mgp.crossval.train_cv(mod,
                                                   Y,
                                                   device,
                                                   train_ps,
                                                   T1=T1,
                                                   N1=N1,
                                                   test=False)
            else:
                # perform crossvalidated training using bGPFA library
                mod, split = mgp.crossval.train_cv_bgpfa(
                    Y,
                    device,
                    train_ps,
                    fit_ts,
                    d_fit,
                    8,
                    T1=T1,
                    N1=N1,
                    test=False,
                    rel_scale=rel_scale,
                    likelihood='Poisson',
                    model='bgpfa',
                    ard=(True if model == 'bgpfa_ard' else False),
                    Bayesian=Bayesian)

            ### need to compute predictive likelihood ###
            Ycv, T1cv, N1cv = split['Y'], split['T1'], split[
                'N1']  # all data, train data and train neurons
            m = T
            T2cv, N2cv = not_in(np.arange(m), T1cv), not_in(
                np.arange(n),
                N1cv)  # these are our training timepoints and neurons

            Ytest = Ycv[:, N2cv, :][...,
                                    T2cv]  #test data (shape ntrial x N2 x T2)

            # extract test latent means (shape ntrial, T2, d)
            if 'GP' in mod.lat_dist.name:  # we used a GP prior over latents
                latents = mod.lat_dist.lat_mu.detach()[:, T2cv, ...]
            else:
                latents = mod.lat_dist.prms[0].detach()[:, T2cv, ...]
            # construct input for prediction of neural activity
            query = latents.transpose(-1, -2).to(device)  #(ntrial, d, m)

            # predicted 'F'
            mu = mod.svgp.predict(query[None, ...], full_cov=False)[0]
            # mean of the implied distribution after transfer function (for test neurons)
            Ypred = detach(mod.svgp.likelihood.dist_mean(mu)[0])[:, N2cv, :]

            # compute log probability of actual test spike counts given test mean predictions
            LL = poisson.logpmf(Ytest, Ypred, loc=0)
            LL = np.mean(LL)  # take avg
            LLs[nrep, dnum, ifold] = LL  # store

            # also compute MSEs
            MSE_vals = np.mean((Ypred - Ytest)**2, axis=(0, -1))
            MSE = np.mean(MSE_vals)  #standard MSE
            MSEs[nrep, dnum, ifold] = MSE

            ######### note that we didn't use the below for anything in the paper ##########

            # also compute training ELBO
            newLLs = []
            for _ in range(100):  # number of runs for memory reasons
                # sample some likelihoods
                svgp_elbo, kl = mod(
                    data,
                    10,
                    m=T,
                    analytic_kl=(False if model == 'fa' else True),
                    neuron_idxs=N1)
                newLLs.append(
                    (svgp_elbo - kl).item() / np.prod(Y.shape) * n /
                    len(N1))  # compute importance weighted log likelihoods
            LL_train = np.mean(newLLs)  # avg across MC samples
            LL_trains[nrep, dnum, ifold] = LL_train  # store result

            # also compute test ELBO
            newMargs = []
            for _ in range(100):
                ##compute marginal ELBO###
                newdat = torch.tensor(Y, device=device)
                svgp_elbo, kl = mod(data[:, :, T2],
                                    10,
                                    batch_idxs=T2,
                                    neuron_idxs=N2,
                                    m=len(T2))
                newMargs.append(
                    (svgp_elbo - kl).item() / (len(T2) * len(N2) * n_samples))
            LL_marg[nrep, dnum, ifold] = np.mean(newMargs)  # store result

            # print some progress
            print(d_fit, 'LL:', np.round(LL, 4), 'MSE:', np.round(MSE, 4),
                  'marg LL:', np.round(LL_marg[nrep, dnum, ifold], 4))

save = True

if save:
    pickle.dump([ds_fit, LLs, LL_marg, MSEs, LL_trains],
                open(savename_cv + '.pickled', 'wb'))
