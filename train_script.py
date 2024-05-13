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

torch.set_default_dtype(torch.float64)

from sklearn.metrics import r2_score
import dill

def main(gen_model_name, rec_model_name, z_path, datapath, dataset='4g10', preprocessor=None, gen_load=False, neural_net=None, noise='Poisson', x_dim=None, train_params=None,
         train_params_rec=None, remove_mean=False, cov_change=False, full_R=False, gen_model_fixed:dict=None,
         data_len=5000, trial_len=100, train_len=4000, CD=1., load_rec_model=None, trained_z=True,
         test_trial_len = 1000, delay=120, generate_random_z=False, held_out_neurons=None):
    if dataset == '4g10':
        model_folder = '4g10datamodels'
    elif dataset == 'Doherty':
        model_folder = 'bgpfa_models'

    torch.manual_seed(0)
    np.random.seed(0)

    torch.cuda.empty_cache()

    if z_path is not None:
        z = np.load(z_path)
        z = torch.Tensor(z)
        print(z.shape, 'z shape')
    else:
        z = None

    if dataset == '4g10':
        z_train = z
        data = np.load(datapath)
        Y_train = data['neural_train']
        Y_train = Y_train.transpose(1,0,2)
        Y_train = torch.Tensor(Y_train)
        print(Y_train.shape, 'y_train shape')

        # Y_test = data['neural_test']
        # Y_test = Y_test.transpose(1,0,2)
        # Y_test = torch.Tensor(Y_test)
        # print(Y_test.shape, 'y_test shape')
    elif dataset == 'Doherty':
        data = pickle.load(open('data/Doherty_example.pickled', 'rb')) # load example data
        binsize = 25 # binsize in ms
        start = 0
        timepoints = np.arange(start, data_len+start) #subsample ~40 seconds of data so things will run somewhat quicker
        # print(data['Y'].shape)
        fit_data = {'Y': data['Y'][..., timepoints], 'locs': data['locs'][timepoints, :], 'targets': data['targets'][timepoints, :], 'binsize': binsize}
        Y = fit_data['Y'] # these are the actual recordings and is the input to our model
        targets = fit_data['targets'] # these are the target locations
        locs = fit_data['locs'] # these are the hand positions

        Y = Y[:, np.mean(Y,axis = (0, 2))/0.025 > 0, :] #subsample highly active neurons so things will run a bit quicker
        # print(Y.shape)
        ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]

        ts = np.arange(Y.shape[-1])*fit_data['binsize'] # measured in ms
        cs = CubicSpline(ts, locs) # fit cubic spline to behavior
        vels = cs(ts+delay, 1) # velocity (first derivative)
        v = Tensor(vels.T[None, ...])
        
        def convert_to_trials(_Y, _z=None, _v=None, t=None):
            assert _Y.shape[-1] % t == 0
            N = _Y.shape[1]
            ntrials = _Y.shape[-1] // t
            Y = _Y.transpose(1,0,2).reshape(N,ntrials,-1).transpose(1,0,2)
            z = None
            if _z is not None:
                b = _z.shape[1]
                z = _z.permute(1,0,2).reshape(b,ntrials,-1).permute(1,0,2)
            v = None
            if _v is not None:
                b_v = _v.shape[1]
                v = _v.permute(1,0,2).reshape(b_v,ntrials,-1).permute(1,0,2)
            return Y, z, v

        print(Y.shape,'original Y shape')
        Y_train = Y[..., :train_len]
        v_train = v[..., :train_len]
        if z is not None:
            z_train = z[..., :train_len]
        else:
            if generate_random_z:
                z = torch.randn((1, p.z_dim, train_len))
            else:
                z = p.get_z_hat(v_train).detach().cpu()
            z_train = z[..., :train_len]
        Y_test = Y[..., train_len:]
        # z_test = z[..., train_len:]
        v_test = v[..., train_len:]
        Y_train, z_train, v_train = convert_to_trials(Y_train, z_train, v_train, trial_len)
        Y_test, _, v_test = convert_to_trials(Y_test, None, v_test, test_trial_len)
        v_test = np.array(v_test)
        Y_train = Tensor(Y_train)
        Y_test = Tensor(Y_test)
        print(Y_train.shape, z_train.shape, v_train.shape, 'y_train, z_train, v_train shape')
        print(Y_test.shape, v_test.shape, 'y_test, v_test shape')
        p.v = v_train.to(device)

    A = preprocessor.A[None, ...].to(device)
    B = torch.linalg.cholesky(preprocessor.Q)[None, ...].to(device)
    mu0 = preprocessor.mu0[None, ...].to(device)
    Sigma0_half = torch.linalg.cholesky(preprocessor.Sigma0)[None, ...].to(device)

    if noise == 'Poisson':
        link_fn = torch.functional.F.softplus
        # link_fn = torch.exp
        lik = Poisson_noise()
    elif noise == 'NB':
        link_fn = lambda x: x
        lik = Negative_binomial_noise(Tensor(Y_train))

    if not gen_load:
        if x_dim is None:
            x_dim = Y_train.shape[1]
            model = LDS(z_train, Y_train, lik, link_fn=link_fn, A=A, B=B, mu0=mu0, Sigma0_half=Sigma0_half, trained_z=trained_z, fixed_d=False, x_dim=x_dim, single_sigma_x=False, full_R=full_R, analytical_init=True)
            model.C.data = torch.eye(model.N).unsqueeze(0).to(device)
            model.C.requires_grad = False
        else:
            model = LDS(z_train, Y_train, lik, link_fn=link_fn, A=A, B=B, mu0=mu0, Sigma0_half=Sigma0_half, trained_z=trained_z, fixed_d=False, x_dim=x_dim, single_sigma_x=False, full_R=full_R)

        print(model.N, model.T, model.x_dim, model.b, model.ntrials)

        if train_params is None:
            train_params = {'batch_size': None, 'n_mc': 50, 'lrate': 5e-3, 'max_steps': 2001, 'step_size': 2000, 'StepLR': True}
        else:
            if 'save_every' in train_params and train_params['save_every'] is not None:
                train_params['save_name'] = model_folder + '/' + gen_model_name
        model.train_supervised_model(model.training_params(**train_params))
        # model.plot_LL()
        model.freeze_params()
        dill.dump(model, open(model_folder + '/' + gen_model_name + '.pkl', 'wb'))
    else:
        model = dill.load(open(model_folder + '/' + gen_model_name + '.pkl', 'rb')).to(device)

    if load_rec_model is not None:
        rec_model = dill.load(open(model_folder + '/' + load_rec_model + '.pkl', 'rb'))
        model = rec_model.gen_model
        preprocessor = rec_model.preprocessor
        rec_model.neural_net.requires_grad = True
    model.freeze_params()
    preprocessor.freeze_params()
    if gen_model_fixed is not None:
        if 'C' in gen_model_fixed or 'all' in gen_model_fixed:
            print('C requires grad')
            model.C.requires_grad = True
        if 'R' in gen_model_fixed or 'all' in gen_model_fixed:
            print('R requires grad')
            if model.full_R:
                model.R_half.requires_grad = True
            else:
                model.log_sigma_x.requires_grad = True
        if 'W' in gen_model_fixed or 'all' in gen_model_fixed:
            print('W requires grad')
            model.W.requires_grad = True
        if 'd' in gen_model_fixed or 'all' in gen_model_fixed:
            print('d requires grad')
            model.d.requires_grad = True
        if 'A' in gen_model_fixed or 'all' in gen_model_fixed:
            print('A requires grad')
            model.A.requires_grad = True
        if 'B' in gen_model_fixed or 'all' in gen_model_fixed:
            print('B requires grad')
            model.B.requires_grad = True
        if 'mu0' in gen_model_fixed or 'all' in gen_model_fixed:
            print('mu0 requires grad')
            model.mu0.requires_grad = True
        if 'Sigma0' in gen_model_fixed or 'all' in gen_model_fixed:
            print('Sigma0 requires grad')
            model.Sigma0_half.requires_grad = True
        if noise == 'NB':
            print('lik requires grad')
            model.lik._total_count.requires_grad = True
        if 'pre_W' in gen_model_fixed or 'all' in gen_model_fixed:
            print('pre_W requires grad')
            preprocessor.W.requires_grad = True
        if 'pre_R' in gen_model_fixed or 'all' in gen_model_fixed:
            print('pre_R requires grad')
            preprocessor.R_half.requires_grad = True
        
    torch.manual_seed(0)
    np.random.seed(0)

    if load_rec_model is None:
        gen_model_fixed_flag = (gen_model_fixed is None)
        rec_model = RecognitionModel(model, rnn=True, neural_net=neural_net, zero_mean_x_tilde=remove_mean, cov_change=cov_change, gen_model_fixed=gen_model_fixed_flag, preprocessor=preprocessor, CD_keep_prob=CD, Y_test=Y_test, v_test=v_test, v_train=v_train, held_out_neurons=held_out_neurons)

    if train_params_rec is None:
        train_params = {'batch_size': 100, 'step_size': 1000, 'lrate': 1e-3, 'max_steps': 1001, 'n_mc_x': 10, 'n_mc_z': 10, 'batch_mc_z': 10}
    else:
        train_params = train_params_rec
        if 'save_every' in train_params and train_params['save_every'] is not None:
            train_params['save_name'] = model_folder + '/' + rec_model_name

    rec_model.train_recognition_model(rec_model.training_params(**train_params))
    # rec_model.plot_LL()

    dill.dump(rec_model, open(model_folder + '/' + rec_model_name + '.pkl', 'wb'))

class MyRNNModel(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

class MyRNNModel2(Module):
    def __init__(self, input_size, hidden_size, output_size1, output_size2):
        super(MyRNNModel2, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        # self.fc_shared = torch.nn.Linear(hidden_size * 2, hidden_size)
        # self.fc_task1 = torch.nn.Linear(hidden_size, output_size1)
        # self.fc_task2 = torch.nn.Linear(hidden_size, output_size2)
        self.fc_task1 = torch.nn.Sequential(
            # torch.nn.Linear(hidden_size * 2, hidden_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_size, output_size1)
            torch.nn.Linear(hidden_size * 2, output_size1)
        )
        self.fc_task2 = torch.nn.Sequential(
            # torch.nn.Linear(hidden_size * 2, hidden_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_size, output_size2)
            torch.nn.Linear(hidden_size * 2, output_size2)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        # out = self.fc_shared(out)
        out_task1 = self.fc_task1(out)
        out_task2 = self.fc_task2(out)
        # out_task2 = torch.tanh(out_task2)
        return torch.cat((out_task1, out_task2), dim=-1)
    
    
class MultiHeadRNN(Module):
    def __init__(self, input_size, hidden_size, output_sizes: dict):
        super(MultiHeadRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.fcs = torch.nn.ModuleDict()
        for key, value in output_sizes.items():
            self.fcs[key] = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, value)
                # torch.nn.Linear(hidden_size * 2, value)
            )
    def forward(self, x):
        out, _ = self.rnn(x)
        ret = dict()
        for key, fc in self.fcs.items():
            ret[key] = fc(out)
        return ret

if __name__ == '__main__':
    # # Doherty
    # p = pickle.load(open('new_params/_1t.pkl', 'rb'))
    # p.freeze_params()
    # train_params = {'batch_size': None, 'n_mc': 100, 'lrate': 5e-2, 'max_steps': 151, 'step_size': 200, 'save_every': 50}
    # train_params_rec = {'batch_size': None, 'step_size': 100, 'lrate': 1e-3, 'max_steps': 501, 'n_mc_x': 10, 'n_mc_z': 10, 'batch_mc_z': 10, 'accumulate_gradient': False, 'save_every': 50}
    # gen_model_fixed = {'C': False, 'R': True, 'W': True, 'd': True}
    # neural_net = None
    # z_path = 'new_params/z_hat_1t_shifted.npy'
    # datapath = None
    # # main('shifted', 'shifted_rec', z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)

    # p = pickle.load(open('new_params/9k1t.pkl', 'rb'))
    # p.freeze_params()
    # train_params = {'batch_size': 64, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 151, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    # train_params_rec = {'batch_size': 8, 'step_size': 30, 'lrate': 1e-3, 'max_steps': 501, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    # # neural_net = MyLSTMModel(200,200,200, bidirectional=False)
    # neural_net = dill.load(open('bgpfa_models/9k_rec.pkl', 'rb')).neural_net.cpu()
    # z_path = 'new_params/z_hat_9k1t_shifted.npy'
    # # main('9k', '9k_rec', data_len=9000, train_len=9000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)

    # p = pickle.load(open('new_params/_1t.pkl', 'rb'))
    # p.freeze_params()
    # neural_net = MyLSTMModel(200,200,200, bidirectional=False)
    # z_path = 'new_params/z_hat_20min_smooth.npy'
    # # main('9ktrain_5kpre', '9ktrain_5kpre_rec', data_len=9000, train_len=9000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)
    # # # Wrong CD, I zeroed out Ys, not the gradients
    # # main('9ktrain_5kpre', '9ktrain_5kpre_rec_CD', data_len=9000, train_len=9000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, CD=0.8)
    # # main('9ktrain_5kpreNB', '9ktrain_5kpre_NB_rec', data_len=9000, train_len=9000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB')
    # # main('9ktrain_5kpreNB', '9ktrain_5kpre_NB_CD_rec', data_len=9000, train_len=9000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # # main('9ktrain_5kpreNB', '9ktrain_5kpre_NB_new_CD_rec', data_len=9000, train_len=9000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # # main('320NB', '320NB_CD_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # # main('320NB', '320NB_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB')
    # # z_path = 'new_params/z_hat_20min_filter.npy'
    # # main('320NB_filt', '320NB_filt_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB')

    # gen_model_fixed = {'C': False, 'R': True, 'W': True, 'd': True, 'A': True, 'B': True}
    # z_path = 'new_params/z_hat_20min_smooth.npy'
    # # main('320NB', '320NB_CD_AB_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # gen_model_fixed = {'C': False, 'R': True, 'W': True, 'd': True, 'A': True, 'B': True, 'pre_W': True, 'pre_R': True}
    # # main('320NB', '320NB_CD_ABp_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8, load_rec_model='320NB_CD_AB_rec')
    
    # # Not training A, C, W etc in recognition model. But A, B are trained in generative model
    # train_params = {'batch_size': 64, 'n_mc': 50, 'lrate': 1e-2, 'max_steps': 501, 'step_size': 20, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    # # main('gen_train', 'gen_train_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8, trained_z=False)

    # train_params_rec = {'batch_size': 8, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 1001, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    # # neural_net = MyLSTMModel(200,100,200, bidirectional=False)
    # # main('320NB', '320NB_CD_ABp_smallrnn_rec', data_len=12800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)


    # # try z_dim = 20
    # p = pickle.load(open('new_params/5k_20z.pkl', 'rb'))
    # gen_model_fixed = {'all': True}
    # neural_net = MyRNNModel(200, 200, 200)
    # z_path = 'new_params/z_hat_5k_20z.npy'
    # train_params = {'batch_size': 64, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 151, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    # train_params_rec = {'batch_size': 8, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    # # Exactly same as '5k_20z_rec'
    # # main('5k_20z', '5k_20z_rec_new', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    
    # p = pickle.load(open('new_params/5k_30z.pkl', 'rb'))
    # z_path = 'new_params/z_hat_5k_30z.npy'
    # # main('5k_30z', '5k_30z_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)

    # p = pickle.load(open('new_params/_1t.pkl', 'rb'))
    # z_path = 'new_params/z_hat_20min_smooth.npy'
    # # main('5k_10z', '5k_10z_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # # main('5k_10z_poisson_noCD', '5k_10z_poisson_noCD_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)
    # # main('5k_10z_poisson_noCD2', '5k_10z_poisson_noCD_rec2', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)
    # # main('5k_10z_NB_noCD', '5k_10z_NB_noCD_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB')

    # # AA2236 THIS WAS RUN ON BIDIRECTIONAL RNN by mistake!!!!!!!
    # # main('5k_10z_NB_CDnew', '5k_10z_NB_CDnew_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)

    # neural_net = MyLSTMModel(200,200,200, bidirectional=False)
    # # main('5k_10z_NB_CDnewLSTM', '5k_10z_NB_CDnewLSTM_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # # main('5k_10z_NB_newLSTM', '5k_10z_NB_newLSTM_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB')
    # train_params_rec = {'batch_size': 8, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 20, 'n_mc_z': 16, 'batch_mc_z': 16, 'accumulate_gradient': False, 'save_every': 10}
    # # main('5k_10z_NB_newLSTM_oldCD', '5k_10z_NB_newLSTM_oldCD_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)
    # # Below is with new CD (scaled yLL)
    # main('5k_10z_NB_newLSTM_oldCD', '5k_10z_NB_newLSTM_newCD_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)

    # z_path = None
    # # AA2236 Below were done with bidirectional RNN I think
    # # main('5k_10z_notshifted', '5k_10z_notshifted_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8, delay=0)
    # # main('5k_10z_notshifted_noCD', '5k_10z_notshifted_noCD_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', delay=0)
    # # main('5k_10z_notshifted_poisson', '5k_10z_notshifted_poisson_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, CD=0.8, delay=0)
    # # main('5k_10z_notshifted_poisson_noCD', '5k_10z_notshifted_poisson_noCD_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, delay=0)


    # # p_new = Preprocessor(p.v.copy(), z_dim=10, W)


    # # train_params = {'batch_size': 16, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 151, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    # # train_params_rec = {'batch_size': 2, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 1001, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    # # neural_net = MyLSTMModel(200,200,200, bidirectional=False)
    # # main('320NB_400t', '320NB_400t_CD_ABp_rec', data_len=12800, train_len=12800, trial_len=400, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', CD=0.8)

    # # # 4g10
    # # p = pickle.load(open('4g10preprocess/10ms.pkl', 'rb'))
    # # p.freeze_params()
    # # train_params = {'batch_size': 200, 'n_mc': 25, 'lrate': 5e-3, 'max_steps': 1001, 'step_size': 2000, 'StepLR': True, 'batch_type': BATCHING.TRIALS, 'save_every': 50}
    # # train_params_rec = {'batch_size': 50, 'step_size': 1000, 'lrate': 1e-3, 'max_steps': 1001, 'n_mc_x': 10, 'n_mc_z': 10, 'batch_mc_z': 10, 'accumulate_gradient': False, 'save_every': 10}
    # # gen_model_fixed = {'C': False, 'R': True, 'W': True, 'd': True}
    # # neural_net = MyLSTMModel(162, 200, 162)
    # # z_path = '4g10preprocess/z_hat_10ms.npy'
    # # datapath = 'data_10ms.npz'

    # # # main('first_10ms', 'first_10ms_rec', z_path=z_path, datapath=datapath, gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)

    # # train_params_rec = {'batch_size': 50, 'step_size': 100, 'lrate': 1e-3, 'max_steps': 1001, 'n_mc_x': 10, 'n_mc_z': 10, 'batch_mc_z': 10, 'accumulate_gradient': False, 'save_every': 10}
    # # neural_net = MyLSTMModel(162, 200, 162, bidirectional=False)
    # # main('first_10ms', 'first_10ms_rec_online', z_path=z_path, datapath=datapath, gen_load=True, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params)

    # Doherty but without preprocess init
    # train_params = {'batch_size': 64, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 0, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    # train_params_rec = {'batch_size': 8, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    train_params = {'batch_size': 50, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 0, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    train_params_rec = {'batch_size': 5, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    fake_v = torch.randn((1,2,2)).to(device)
    p = Preprocessor(v=fake_v, z_dim=10, noise_scale=0.1)
    datapath = None
    z_path = None
    gen_model_fixed = {'all': True}
    neural_net = MyLSTMModel(180,200,200, bidirectional=False)

    np.random.seed(0)
    indices = np.random.choice(200, 20, replace=False)
    print(indices)
    # main('no_init', 'no_init_rec', data_len=15000, train_len=5000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', held_out_neurons=indices, generate_random_z=True)

    train_params_rec = {'batch_size': 5, 'step_size': 50, 'lrate': 1e-2, 'max_steps': 1001, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    # main('no_init2', 'no_init2_rec', data_len=15000, train_len=5000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', held_out_neurons=indices, generate_random_z=True)

    train_params = {'batch_size': 25, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 1, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    train_params_rec = {'batch_size': 5, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10}
    p = pickle.load(open('new_params/_1t.pkl', 'rb'))
    z_path = 'new_params/z_hat_20min_smooth.npy'
    # main('NB_co', 'NB_co_rec', data_len=10000, train_len=5000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', held_out_neurons=indices)
    
    train_params = {'batch_size': 64, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 1, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    train_params_rec = {'batch_size': 8, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 18, 'n_mc_z': 18, 'batch_mc_z': 18, 'accumulate_gradient': False, 'save_every': 10, 'test_co_smoothing_samps': 0}
    # main('NB_co_long', 'NB_co_long_rec', data_len=42800, train_len=12800, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', held_out_neurons=indices)



    train_params = {'batch_size': 25, 'n_mc': 50, 'lrate': 5e-2, 'max_steps': 151, 'step_size': 200, 'save_every': 50, 'batch_type': BATCHING.TRIALS}
    train_params_rec = {'batch_size': 5, 'step_size': 50, 'lrate': 1e-3, 'max_steps': 301, 'n_mc_x': 20, 'n_mc_z': 20, 'batch_mc_z': 20, 'accumulate_gradient': False, 'save_every': 10, 'test_co_smoothing_samps': 100, 'test_co_smoothing_samps_per_batch': 10}
    neural_net = MyLSTMModel(180,180,200, bidirectional=True)
    # main('NB_co_bi', 'NB_co_bi_rec', data_len=10000, train_len=5000, trial_len=100, z_path=z_path, datapath=datapath, dataset='Doherty', gen_load=False, full_R=True, x_dim=None, neural_net=neural_net, preprocessor=p, gen_model_fixed=gen_model_fixed, train_params_rec=train_params_rec, train_params=train_params, noise='NB', held_out_neurons=indices)
