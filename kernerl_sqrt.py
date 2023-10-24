import abc
import torch
from torch import Tensor
import numpy as np

class Kernel_sqrt(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def k_1_2():
        pass

    def full_matrix():
        pass

class SquaredExponential(Kernel_sqrt):
    def k_1_2(omega, l):
        prefactor =  ((2*np.pi)**(1/4)) * (l**(-1/2))
        return prefactor * np.exp(-((omega*l)**2) / 4)
        