"""
utils/param.py

Implements `Param` object, a helper class for specifying the parameters of a
model instance.
"""
import os
from functools import reduce
from math import pi as PI
import torch

class Param:
    def __init__(
        self,
        beta: float = 6.0,
        lat: list = [64, 64],
        tau: float = 2.0,
        nstep: int = 50,
        ntraj: int = 256,
        nrun: int = 4,
        nprint: int = 256,
        seed: int = 11*13,
        randinit: bool = False,
        nth: int = int(os.environ.get('OMP_NUM_THREADS', '2')),
        nth_interop: int = 2
    ):
        self.beta = beta
        self.lat = lat
        self.nd = len(lat)
        self.volume = reduce(lambda x, y: x * y, lat)
        self.tau = tau
        self.nstep = nstep
        self.dt = self.tau / self.nstep
        self.ntraj = ntraj
        self.nrun = nrun
        self.nprint = nprint
        self.seed = seed
        self.randinit = randinit
        self.nth = nth
        self.nth_interop = nth_interop

    def initializer(self):
        if self.randinit:
            return torch.empty((self.nd,) + self.lat).uniform_(-PI, PI)
        else:
            return torch.zeros((self.nd,) + self.lat)

    def summary(self):
        status = {
            'latsize': self.lat,
            'volume': self.volume,
            'beta': self.beta,
            'trajs': self.ntraj,
            'tau': self.tau,
            'steps': self.nstep,
            'seed': self.seed,
            'nth': self.nth,
            'nth_interop': self.nth_interop,
        }

        #  return ', '.join('='.join((str(k), str(v))) for k, v in status.items())
        s = ', '.join('='.join((str(k), str(v))) for k, v in status.items())
        return f'{s}\n'

    def uniquestr(self):
        lat = ".".join(str(x) for x in self.lat)
        return (
            f'out_l{lat}_b{self.beta}_n{self.ntraj}'
            f'_t{self.tau}_s{self.nstep}.out'
        )
