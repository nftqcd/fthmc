"""
config.py

Contains various utilities used throughout the project.
"""
from __future__ import absolute_import, annotations, print_function

import os
from dataclasses import dataclass, field
from math import pi as PI
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.io as io

__author__ = 'Sam Foreman'
__date__ = '05/23/2021'


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

GREEN = '#87ff00'
PINK = '#F92672'
BLUE = '#007dff'
RED = '#ff4050'



NOW = io.get_timestamp('%Y-%m-%d-%H%M%S')
METRIC_NAMES = ['dt', 'accept', 'traj', 'dH', 'expdH', 'plaq', 'charge']

logger = io.Logger()
TWO_PI = 2 * PI


ActionFn = Callable[[float], torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()

def list_to_arr(x: list):
    return np.array([grab(torch.stack(i)) for i in x])


from fthmc.utils.param import Param

@dataclass
class qedMetrics:
    param: Param
    action: torch.Tensor
    plaq: torch.Tensor
    charge: torch.Tensor

    def __post_init__(self):
        self._metrics = {
            'action': self.action,
            'plaq': self.plaq,
            'charge': self.charge
        }


@dataclass
class ftMetrics:
    force_norm: torch.Tensor
    ft_action: torch.Tensor
    p_norm: torch.Tensor


@dataclass
class State:
    x: torch.Tensor
    p: torch.Tensor


@dataclass
class TrainConfig:
    n_era: int = 10
    n_epoch: int = 100
    n_layers: int = 24
    n_s_nets: int = 2
    hidden_sizes: list[int] = field(default_factory=lambda: [8, 8])
    kernel_size: int = 3
    base_lr: float = 0.001
    batch_size: int = 64
    print_freq: int = 10
    plot_freq: int = 20
    with_force: bool = False




@dataclass
class PlotObject:
    #  fig: plt.Figure
    ax: plt.Axes
    line: list[plt.Line2D]


@dataclass
class LivePlotData:
    data: Any
    plot_obj: PlotObject
