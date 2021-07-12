"""
samplers.py
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Callable, Union

import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from fthmc.utils.distributions import calc_dkl, calc_ess
import fthmc.utils.qed_helpers as qed

import torch.nn as nn
from fthmc.utils.distributions import bootstrap, BasePrior
from fthmc.utils.logger import Logger
from fthmc.config import DTYPE, FlowModel


#  NumpyFloat = Union[np.float64, np.float32, np.float64]
#  NumpyFlow = np.float
NumpyObject = Union[np.ndarray, np.float]
TensorLike = Union[list, NumpyObject, torch.Tensor]

logger = Logger()


def grab(x: TensorLike):
    #  if isinstance(x, (np.ndarray, np.float32, np.float64)):
    #      return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.array(x)


def list_to_arr(x: list):
    return np.array([grab(torch.stack(i)) for i in x])


def apply_flow_to_prior(
        prior: BasePrior,
        coupling_layers: nn.ModuleList,
        *,
        batch_size: int,
        xi: torch.Tensor = None
):
    if xi is None:
        xi = prior.sample_n(batch_size)

    x = xi.clone().to(DTYPE)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq = logq - logJ

    return x, xi, logq


def apply_flow_to_prior1(
        prior: BasePrior,
        coupling_layers: list[nn.Module],
        *,
        batch_size: int,
        xi: torch.Tensor = None
):
    if xi is None:
        xi = prior.sample_n(batch_size)

    x = xi
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logdet = layer.forward(x)
        logq = logq - logdet

    return x, xi, logq


ActionFn = Callable[[float], torch.Tensor]

def generate_ensemble(
        model: nn.Module,
        action: ActionFn,
        ensemble_size: int = 1024,
        batch_size: int = 64,
        nboot: int = 100,
        binsize: int = 16,
):
    """Calculate the topological susceptibility by generating an ensemble."""
    history = make_mcmc_ensemble(model, action, batch_size, ensemble_size)
    qarr = np.array([grab(i) for i in history['q']])
    qsq_mean, qsq_err = bootstrap(qarr ** 2, nboot=nboot, binsize=binsize)
    #  charge = grab(qed.topo_charge(torch.stack(ensemble['x'], dim=0)))
    #  xmean, xerr = bootstrap(charge ** 2, nboot=nboot, binsize=binsize)

    logger.log(f'accept_rate={np.mean(history["accepted"])}')
    logger.log(f'top_susceptibility={qsq_mean:.5f} +/- {qsq_err:.5f}')

    return {
        'history': history,
        'suscept_mean': qsq_mean,
        'suscept_err': qsq_err,
    }



def serial_sample_generator_old(
        model: FlowModel,
        action_fn: ActionFn,
        batch_size: int,
        num_samples: int
):
    prior = model.prior
    layers = model.layers
    layers.eval()
    #  model.layers.eval()
    x, q, logq, logp = None, None, None, None
    for i in range(num_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            _, x, logq = apply_flow_to_prior(prior, layers,
                                             batch_size=batch_size)
            logp = -action_fn(x)
            q = qed.batch_charges(x)

        yield x[batch_i], q[batch_i], logq[batch_i], logp[batch_i]


def serial_sample_generator(model, action, batch_size, N_samples):
    if isinstance(model, dict):
        layers, prior = model['layers'], model['prior']
    elif isinstance(model, FlowModel):
        layers, prior = model.layers, model.prior

    layers.eval()
    x, logq, logp = None, None, None
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            _, x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
            logp = -action(x)
        yield x[batch_i], logq[batch_i], logp[batch_i]


def make_mcmc_ensemble_new(model, action, batch_size, N_samples):
    history = {
        'x' : [],
        'logq' : [],
        'logp' : [],
        'accepted' : []
    }

    # build Markov chain
    sample_gen = serial_sample_generator(model, action, batch_size, N_samples)
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:
            # always accept first proposal, Markov chain must start somewhere
            accepted = True
        else:
            # Metropolis acceptance condition
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
            p_accept = min(1, p_accept)
            draw = torch.rand(1) # ~ [0,1]
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logp = last_logp
                new_logq = last_logq
        # Update Markov chain
        history['logp'].append(new_logp)
        history['logq'].append(new_logq)
        history['x'].append(new_x)
        history['accepted'].append(accepted)

    return history

def make_mcmc_ensemble(
    model,
    action_fn,
    batch_size,
    num_samples,
    writer: SummaryWriter = None
):
    xarr = []  # for holding the configurations
    names = ['x', 'q', 'dqsq', 'logq', 'logp', 'acc']
    history = {
        name: [] for name in names
    }

    # Build Markov chain
    sample_gen = serial_sample_generator(model, action_fn, batch_size,
                                         num_samples)
    step = 0
    with torch.no_grad():
        #  for x_new, q_new, logq_new, logp_new in sample_gen:
        for x_new, logq_new, logp_new in sample_gen:
            # Always accept the first proposal
            if len(history['logp']) == 0:
                accepted = True
                q_old = qed.topo_charge(x_new[None, :])
            else:
                q_old = qed.topo_charge(xarr[-1][None, :])
                logp_old = history['logp'][-1]
                logq_old = history['logq'][-1]
                p_accept = torch.exp(
                    (logp_new - logq_new) - (logp_old - logq_old)
                )
                p_accept = min(1, p_accept)
                draw = torch.rand(1)  # ~ [0, 1]
                if draw < p_accept:
                    accepted = True
                else:
                    accepted = False
                    #  x_new = history['x'][-1]
                    x_new = xarr[-1]
                    #  q_new = q_old
                    logp_new = logp_old
                    logq_new = logq_old

            q_new = qed.topo_charge(x_new[None, :])
            #  q_new = qed.batch_charges(x_new[None, :])
            # Update Markov Chain
            xarr.append(x_new)
            metrics = {
                'q': q_new,
                'dqsq': (q_new - q_old) ** 2,
                #  'dkl': calc_dkl(logp_new, logq_new),
                #  'ess': calc_ess(logp_new, logq_new),
                'logp': logp_new,
                'logq': logq_new,
                'acc': float(accepted),
            }

            for key, val in metrics.items():
                try:
                    history[key].append(val)
                except KeyError:
                    history[key] = [val]

                if writer is not None:
                    v = torch.tensor(val)
                    if len(v.shape) > 1:
                        writer.add_histogram(f'inference/{key}', v,
                                             global_step=step)
                    else:
                        writer.add_scalar(f'inference/{key}', v.mean(),
                                          global_step=step)

            step += 1

    history_ = {
        k: torch.Tensor(v).cpu().numpy() for k, v in history.items()
    }
    return history_
