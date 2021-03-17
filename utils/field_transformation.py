"""
utils/field_transformation.python

Contains various functions from

https://arxiv.org/abs/2101.08176

"Introduction to Normalizing Flows for Lattice Field Theory
Michael S. Albergo, Denis Boyda, Daniel C. Hackett, Gurtej Kanwar, Kyle
Cranmer, Sébastien Racanière, Danilo Jimenez Rezende, Phiala E. Shanahan"
License: CC BY 4.0

with slight modifications by Xiao-Yong Jin to reduce global variables
"""
import math

from timeit import default_timer as timer
from functools import reduce

import numpy as np
import packaging.version

import torch

from utils.distributions import MultivariateUniform

TORCH_DEVICE = 'cpu'
FLOAT_DTYPE = np.float64

# pylint:disable=missing-function-docstring

def torch_mod(x: torch.Tensor):
    return torch.remainder(x, 2*np.pi)


def torch_wrap(x: torch.Tensor):
    return torch_mod(x+np.pi) - np.pi


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def compute_ess(logp: torch.Tensor, logq: torch.Tensor):
    logw = logp - logq
    log_ess = (2 * torch.logsumexp(logw, dim=0)
               - torch.logsumexp(2*logw, dim=0))
    ess_per_cfg = torch.exp(log_ess) / len(logw)

    return ess_per_cfg


def bootstrap(x: np.ndarray, *, nboot: int, binsize: int):
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    boots = np.array([
        np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0, 1))
        for _ in range(nboot)
    ])
    #  for _ in range(Nboot):
    #      boots.append(
    #          np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0, 1))
    #      )
    return np.mean(boots), np.std(boots)


def print_metrics(history, avg_last_N_epochs, era, epoch):
    print(f'== Era {era} | Epoch {epoch} metrics ==')
    for key, val in history.items():
        avgd = np.mean(val[-avg_last_N_epochs:])
        print(f'{key}: {avgd:g}')
    print('\n' + 80 * '=' + '\n')


def serial_sample_generator(model, action, batch_size, N_samples):
    layers, prior = model['layers'], model['prior']
    layers.eval()
    x, logq, logp = None, None, None
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            xi, x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
            logp = -action(x)
        yield x[batch_i], logq[batch_i], logp[batch_i]


def make_mcmc_ensemble(model, action, batch_size, N_samples):
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


def apply_flow_to_prior(prior, coupling_layers, *, batch_size, xi = None):
    if xi == None:
        xi = prior.sample_n(batch_size)
    x = xi
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq = logq - logJ
    return xi, x, logq


def compute_u1_plaq(links, mu, nu):
    """Compute U(1) plaqs in the (mu,nu) plane given `links` = arg(U)"""
    return (links[:,mu] + torch.roll(links[:,nu], -1, mu+1)
            - torch.roll(links[:,mu], -1, nu+1) - links[:,nu])


class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu+1,Nd):
                action_density = action_density + torch.cos(
                    compute_u1_plaq(cfgs, mu, nu))
        return -self.beta * torch.sum(action_density, dim=tuple(range(1,Nd+1)))


def gauge_transform(links, alpha):
    for mu in range(len(links.shape[2:])):
        links[:,mu] = alpha + links[:,mu] - torch.roll(alpha, -1, mu+1)
    return links


def random_gauge_transform(x):
    Nconf, VolShape = x.shape[0], x.shape[2:]
    return gauge_transform(x, 2*np.pi*torch.rand((Nconf,) + VolShape))


def topo_charge(x):
    P01 = torch_wrap(compute_u1_plaq(x, mu=0, nu=1))
    axes = tuple(range(1, len(P01.shape)))
    return torch.sum(P01, dim=axes) / (2*np.pi)


