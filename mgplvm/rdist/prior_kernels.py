import torch
import numpy as np


def k_1_2_squared_exponential(f, l):
    omega = 2 * np.pi * f
    prefactor = ((2 * np.pi)**(1 / 4)) * (l**(-1 / 2))
    return prefactor * torch.exp(-((omega * l)**2) / 4)


def k_1_2_exponential(f, l):
    pass


def k_1_2_matern(f, l):
    pass
