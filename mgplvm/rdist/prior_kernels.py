import torch
import numpy as np


def k_1_2_squared_exponential(f, l):
    omega = 2 * np.pi * f
    prefactor = ((2 * np.pi)**(1 / 4)) * (l**(-1 / 2))
    return prefactor * torch.exp(-((omega * l)**2) / 4)


def k_1_2_exponential(f, l):
    omega = 2 * np.pi * f
    return torch.sqrt(1/(omega + 1/l))


def k_1_2_matern_3_2(f, l):
    omega = 2 * np.pi * f
    a = np.sqrt(3)/l
    fractional_term = 1/(omega + a)
    fourier = fractional_term + a * torch.square(fractional_term)
    return torch.sqrt(fourier)
