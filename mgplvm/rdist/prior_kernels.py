import torch
import numpy as np


def k_1_2_squared_exponential(f, l):
    omega = 2 * np.pi * f

    # prefactor = ((2 * np.pi)**(1 / 4)) * (l**(-1 / 2))
    # return prefactor * torch.exp(-((omega * l)**2) / 4)

    prefactor = ((2 * np.pi)**(1 / 4)) * torch.sqrt(l)
    return prefactor * torch.exp(-((omega * l)**2) / 4)


def k_1_2_exponential(f, l):
    omega = 2 * np.pi * f
    fourier = (2*l) / (1 + (omega * l)**2)
    return torch.sqrt(fourier)


def k_1_2_matern_3_2(f, l):
    omega = 2 * np.pi * f
    # a = np.sqrt(3)/l
    # fractional_term = 1/(omega + a)
    # fourier = fractional_term + a * torch.square(fractional_term)
    # return torch.sqrt(fourier)

    # (1+(sqrt(3)*|t|)/l)e^(-(sqrt(3)*|t|)/l)

    num = torch.sqrt(12*np.sqrt(3)*l)
    denom = 3 + torch.square(l*omega)
    return num/denom

def k_1_2_matern_5_2(f, l):
    omega = 2 * np.pi * f

    num = torch.sqrt(400*np.sqrt(5)*l)
    denom = np.sqrt(3)*torch.pow(torch.square(l*omega) + 5, 1.5)
    return num/denom
