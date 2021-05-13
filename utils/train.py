"""
train.py

Contains helper functions for training Flow-based HMC.
"""
from __future__ import absolute_import, print_function, division, annotations
import os
import torch
import numpy as np
import time
import datetime
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from typing import Callable

import .qed_helpers as qed
from .param import Param
from .layers import make_u1_equiv_layers, set_weights
from .distributions import MultivariateUniform, bootstrap, calc_dkl, calc_ess
from .samplers import make_mcmc_ensemble, apply_flow_to_prior

try:
    from rich.theme import Theme
    from rich.console import Console
    from rich.style import Style
    theme = Theme({
        'repr.number': 'bold bright_green',
        'repr.attrib_name': 'bold bright_magenta',
    })
    console = Console(record=False, log_path=False, width=256,
                      log_time_format='[%X] ', theme=theme)
    def put(s):
        console.log(s)
except (ImportError, ModuleNotFoundError):
    def put(s):
        print(s)


@dataclass
class ModelDict:
    layers: nn.Module
    prior: nn.Module


def print_metrics(
        history: dict[str, torch.Tensor],
        avg_last_n_epochs: int = 10,
        era: int = None,
        epoch: int = None
):
    outstr = []
    if era is not None:
        outstr.append(f'era: {era}')
    if epoch is not None:
        outstr.append(f'epoch: {epoch}')
    for key, val in history.items():
        val = np.array(val)
        if len(val.shape) > 0:
            avgd = np.mean(val[-avg_last_n_epochs:])
        else:
            avgd = np.mean(val)
        outstr.append(f'{key}: {avgd:g}')
    outstr = ', '.join(outstr)
    put(outstr)

    return outstr



def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def run(param: Param, field: torch.Tensor, verbose: bool = False):
    if field is None:
        field = param.initializer()

    fields = {i: [] for i in range(param.nrun)}
    metrics = {
        'dt': [],
        'accept': [],
        'traj': [],
        'dH': [],
        'expdH': [],
        'plaq': [],
        'charge': [],
    }
    #  metrics = {i: metric for i in range(param.nrun)}
    now = get_timestamp('%Y-%m-%d')
    outdir = os.path.join(os.getcwd(), now)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    outfile = os.path.join(outdir, param.uniquestr())
    with open(outfile, 'w') as f:
        params = param.summary()
        f.write(params)
        put(params)
        plaq = qed.action(param, field)
        charge = qed.topo_charge(field[None, :])
        status = f'Initial configuration: plaq: {plaq}, charge: {charge}\n'
        f.write(status)
        put(status)
        for n in range(param.nrun):
            t0 = time.time()
            for i in range(param.ntraj):
                dH, expdH, acc, field = qed.hmc(param, field, verbose=verbose)
                plaq = qed.action(param, field) / (-param.beta * param.volume)
                charge = qed.topo_charge(field[None, :])
                ifacc = 'ACCEPT' if acc else 'REJECT'
                batch_metrics = {
                    #  'dt': time.time() - t0,
                    'accept': bool(acc),
                    'traj': n * param.ntraj + i + 1,
                    'dH': dH,
                    'expdH': expdH,
                    'plaq': plaq,
                    'charge': int(qed.grab(charge)[0]),
                }
                status = {
                    k: f'{v:<12.8g}' if isinstance(v, float)
                    else f'{v:<4}' for k, v in batch_metrics.items()
                }
                outstr = ', '.join(
                    '='.join((k, v)) for k, v in status.items()
                )
                f.write(outstr + '\n')

                if (i + 1) % (param.ntraj // param.nprint) == 0:
                    put(outstr)

                fields[n] = field

            batch_metrics['dt'] = time.time() - t0
            for key, val in batch_metrics.items():
                metrics[key].append(val)

        dt = metrics['dt']
        put(f'Run times: {dt}')
        put(f'Per trajectory: {[t / param.ntraj for t in dt]}')

    return fields, metrics


def train_step(
    model: ModelDict,
    action: Callable[torch.Tensor, torch.Tensor],
    optimizer: optim.Optimizer,
    metrics: dict,
    batch_size: int,
    with_force: bool = False,
    pre_model: ModelDict = None,
    verbose: bool = True,
    era: int = None,
    epoch: int = None
):
    """Perform a single training step."""
    t0 = time.time()
    layers = model
    layers. prior = model.layers, model.prior
    optimizer.zero_grad()

    xi = None
    if pre_model != None:
        pre_layers = pre_model.layers
        pre_prior = pre_model.prior
        pre_xi = pre_prior.sample_n
