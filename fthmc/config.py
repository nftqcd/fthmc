"""
config.py

Contains definitions of `Param` object specifying parameters of the lattice,
and `TrainConfig` object specifying parameters of the training run.
"""
from __future__ import absolute_import, annotations, print_function
from functools import reduce

import os
from dataclasses import dataclass, field
from math import pi as PI
from typing import Callable

#  import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import json

__author__ = 'Sam Foreman'
__date__ = '05/23/2021'


npDTYPE = np.float32
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    DEVICE = 'cpu'
    torch.set_default_tensor_type(torch.FloatTensor)


from fthmc.utils.logger import Logger, get_timestamp
from fthmc.utils.distributions import BasePrior

PLAQ_EXACT = {
    1.0: 0.44638990, 1.5: 0.59613320,
    2.0: 0.69777477, 2.5: 0.76499665,
    3.0: 0.80998540, 3.5: 0.84110373,
    4.0: 0.86352290, 4.5: 0.88033150,
    5.0: 0.89338326, 5.5: 0.90381753,
    6.0: 0.91235965, 6.5: 0.91948840,
    7.0: 0.92553246, 7.5: 0.93072510,
    8.0: 0.93523590, 8.5: 0.93919160,
    9.0: 0.94268996, 9.5: 0.94580620
}

LATEX_STRS = {
    'q': r"$Q$",
    'dq': r"$\delta Q$",
    'plaq': r"$x_{P}$",
    'acc': r"$A(x'|x)$",
    'dh': r"$\delta H$",
    'dt': r"$\delta t$",
    'logp': r"$\log(p)$",
    'logq': r"$\log(q)$",
}

logger = Logger()
DTYPE = torch.get_default_dtype()

DPI = 150
FIGSIZE = (9, 3)
NUM_SAMPLES = 8192
CHAINS_TO_PLOT = 4
THERM_FRAC = 0.2
KWARGS = {
    'dpi': DPI,
    'figsize': FIGSIZE,
    'num_samples': NUM_SAMPLES,
    'chains_to_plot': CHAINS_TO_PLOT,
    'therm_frac': THERM_FRAC,
}


logger.log(f'TORCH DEVICE: {DEVICE}')
logger.log(f'TORCH DTYPE: {DTYPE}')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

RED = '#FF4050'
BLUE = '#007DFF'
PINK = '#F92672'
GREEN = '#87ff00'
YELLOW = '#FFFF00'

NOW = get_timestamp('%Y-%m-%d-%H%M%S')
METRIC_NAMES = ['dt', 'accept', 'traj', 'dH', 'expdH', 'plaq', 'charge']

TWO_PI = 2 * PI


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def list_to_arr(x: list):
    return np.array([grab(torch.stack(i)) for i in x])


# TODO: Split up TrainConfig using NetworkConfig, StepsConfig, ExperimentConfig

@dataclass
class FlowModel:
    prior: BasePrior
    layers: torch.nn.ModuleList


@dataclass
class ExperimentConfig:
    L: int          # lattice Volume
    beta: float     # coupling constant


@dataclass
class NetworkConfig:
    activation_fn: str = 'silu'
    batch_size: int = 64
    base_lr: float = 0.001
    n_s_nets: int = 2            # Number of (RealNVP) coupling layers, `s_net`
    n_layers: int = 24           # Number of hidden layers in each `s_net`
    kernel_size: int = 3         # Kernel size in Conv2D layers
    with_force: bool = False     # Minimize force norm during training
    # hidden layers between convolutional layers
    hidden_sizes: list[int] = field(default_factory=lambda: [8, 8])


@dataclass
class StepsConfig:
    n_era: int = 10
    n_epoch: int = 100
    print_freq: int = 100
    plot_freq: int = 100
    log_freq: int = 100

@dataclass
class BaseHistory:
    skip: list[str] = field(default_factory=list)

    def update(self, metrics: dict[str, torch.Tensor]):
        for key, val in metrics.items():
            if key in self.skip:
                continue
            try:
                self.__dict__[key].append(val)
            except KeyError:
                self.__dict__[key] = [val]


@dataclass
class SchedulerConfig:
    factor: float
    mode: str = 'min'
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = 'rel'
    cooldown: int = 0
    min_lr: float = 1e-5
    verbose: bool = True

    def __repr__(self):
        status = {k: v for k, v in self.__dict__.items()}
        s = '\n'.join('='.join((str(k), str(v))) for k, v in status.items())
        return '\n'.join(['Param:', 16 * '-', s])

    def to_json(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return attrs

    def summary(self):
        return self.__repr__

    def titlestr(self):
        tstr = ', '.join([f'factor: {self.factor}',
                          f'patience: {self.patience}',
                          f'min_lr: {self.min_lr}'])
        return tstr

    def uniquestr(self):
        return '_'.join([f'f{self.factor}',
                         f'p{self.patience}',
                         f'm{self.min_lr}'])


@dataclass
class Param:
    beta: float = 6.0       # Inverse coupling constant
    L: int = 8              # Linear extent of square lattice, (L x L)
    tau: float = 2.0        # Trajectory length
    nstep: int = 10         # Number of leapfrog steps / trajectory
    ntraj: int = 256        # Number of trajectories to generate for HMC
    nrun: int = 4           # Number of indep. HMC experiments to run
    nprint: int = 256       # How frequently to print metrics during HMC
    seed: int = 11 * 13     # Random seed
    randinit: bool = False  # Start from randomly initialized configuration?
    # nth_interop: int = 2    # Number of interop threads
    # nth: int = int(os.environ.get('OMP_NUM_THREADS', '2'))  # n OMP threads

    def __post_init__(self):
        self.lat = [self.L, self.L]
        self.nd = len(self.lat)
        self.shape = [self.nd, *self.lat]
        #  self.shape = [self.batch_size, self.nd, *self.lat]
        self.volume = reduce(lambda x, y: x * y, self.lat)
        self.dt = self.tau / self.nstep

        basedir = os.path.join(LOGS_DIR, 'hmc')
        lat = "x".join(str(x) for x in self.lat)
        self.logdir = os.path.join(
            basedir,
            f'lat{lat}',
            f'beta{self.beta}',
            self.uniquestr()
        )

    def initializer(self):
        if self.randinit:
            x = torch.empty([self.nd, ] + self.lat).uniform_(-PI, PI)
        else:
            x = torch.zeros([self.nd, ] + self.lat)

        return x[None, :]

    def __repr__(self):
        status = {k: v for k, v in self.__dict__.items() if k != 'dirs'}
        s = '\n'.join('='.join((str(k), str(v))) for k, v in status.items())
        return '\n'.join(['Param:', 16 * '-', s])

    def to_json(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return attrs

    def summary(self):
        return self.__repr__

    def titlestr(self):
        return ', '.join([r'$\times$'.join(str(x) for x in self.lat),
                          fr"""$\beta: {self.beta}$""",
                          fr"""$\tau: {self.tau}$""",
                          fr"""$\ell: {self.nstep}$""",
                          fr"""$\delta t: {self.dt}$"""])

    def uniquestr(self):
        lat = "x".join(str(x) for x in self.lat)
        return '_'.join([f't{lat}',
                         f'b{self.beta}',
                         f'n{self.ntraj}',
                         f't{self.tau}',
                         f's{self.nstep}'])

@dataclass
class lfConfig:
    tau: float  # trajectory length
    nstep: int  # number of leapfrog steps per trajectory

    def __post_init__(self):
        self.dt = self.tau / self.nstep  # step size

    def __repr__(self):
        status = {k: v for k, v in self.__dict__.items() }
        return json.dumps(status, indent=4)

    def titlestr(self):
        return ', '.join([
            rf"""$\tau: {self.tau}$""",
            rf"""$\ell: {self.nstep}$""",
            rf"""$\delta t: {self.dt}$""",
        ])

    def uniquestr(self):
        return '_'.join([f't{self.tau}', f's{self.nstep}', f'dt{self.dt}'])


@dataclass
class TrainConfig:
    L: int
    beta: float
    restore: bool = False
    activation_fn: str = 'silu'  # activation to use in ConvNet
    n_era: int = 10              # Each `era` consists of `n_epoch` epochs
    n_epoch: int = 100           # Number of `epochs` (loss + backprop)
    batch_size: int = 64         # Number of chains to maintain in parallel
    base_lr: float = 0.001       # Base learning rate
    n_s_nets: int = 2            # Number of (RealNVP) coupling layers, `s_net`
    n_layers: int = 24           # Number of hidden layers in each `s_net`
    kernel_size: int = 3         # Kernel size in Conv2D layers
    with_force: bool = False     # Minimize force norm during training
    print_freq: int = 50         # How frequently to print training metrics
    plot_freq: int = 50          # How frequently to update training plots
    log_freq: int = 50           # How frequently to log TensorBoard summaries
    debug: bool = False          # Treat as a debug run, dont clutter stats
    # Sizes of hidden layers between convolutional layers
    hidden_sizes: list[int] = field(default_factory=lambda: [8, 8])

    def update_logdirs(self, logdir: str):
        self.logdir = logdir
        dtrain = os.path.join(self.logdir, 'training')
        dinfer = os.path.join(self.logdir, 'inference')
        dckpts = os.path.join(dtrain, 'checkpoints')
        dirs = {
            'logdir': self.logdir,
            'training': dtrain,
            'inference': dinfer,
            #  'plots': dplots,
            'ckpts': dckpts,
        }
        # TODO [DDP/Horovod]: only create dirs if rank == 0
        for _, d in dirs.items():
            if not os.path.isdir(d):
                os.makedirs(d)

        self.dirs = dirs

        return dirs

    def __post_init__(self):
        # ------------------------------------------------------------
        # TODO:
        # - Deal with batch of configurations
        #     self.shape = [self.batch_size, self.nd, *self.lat]
        # ------------------------------------------------------------
        self.lat = [self.L, self.L]
        self.nd = len(self.lat)
        self.shape = [self.nd, *self.lat]
        self.latstr = "x".join(str(x) for x in self.lat)
        self.volume = reduce(lambda x, y: x * y, self.lat)
        ustr = [f'lat{self.latstr}', f'beta{self.beta}', self.uniquestr()]
        if self.debug:
            here = os.getcwd()
            logger.log(f'Using working directory for logs!\n   basedir={here}')
            basedir = os.path.join(here, 'debug')
        else:
            basedir = os.path.join(LOGS_DIR, 'models')

        logdir = os.path.join(basedir, *ustr)
        self.dirs = self.update_logdirs(logdir)

    def titlestr(self):
        hstr = '[' + ','.join([f'{i}' for i in self.hidden_sizes]) + ']'
        lstr = 'x'.join(str(x) for x in self.lat)
        return ', '.join([f'{lstr} ',
                          fr'$\beta: {self.beta}$'
                          fr'$B: {self.batch_size}$',
                          fr'\sigma: {self.activation_fn}$',
                          fr'$N: {self.n_layers}$',
                          fr'$M: {self.n_s_nets}$',
                          fr'$W: {self.kernel_size}$',
                          fr'$n: {hstr}$',
                          fr'lr: {self.base_lr}$'])

    def uniquestr(self):
        hstr = ''.join([f'{i}' for i in self.hidden_sizes])
        return '_'.join([f'L{self.L}',
                         f'b{self.beta}',
                         f'nb{self.batch_size}',
                         f'act{self.activation_fn}',
                         f'nh{self.n_layers}',
                         f'ns{self.n_s_nets}',
                         f'ks{self.kernel_size}',
                         f'hl{hstr}',
                         f'lr{self.base_lr}',
                         f'era{self.n_era}',
                         f'epoch{self.n_epoch}'])

    def __repr__(self):
        # status = {k: v for k, v in self.__dict__.items() if k != 'dirs'}
        status = {k: v for k, v in self.__dict__.items() }
        return json.dumps(status, indent=4)


    def to_json(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return attrs

    def summary(self):
        return self.__repr__
