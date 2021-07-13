"""
utils/qed_helpers.py

Contains various helper functions for training the normalizing flow model on a
two-dimensional U(1) (QED) theory.

# U1GaugeAction taken from:
https://arxiv.org/abs/2101.08176

"Introduction to Normalizing Flows for Lattice Field Theory
Michael S. Albergo, Denis Boyda, Daniel C. Hackett, Gurtej Kanwar, Kyle
Cranmer, Sébastien Racanière, Danilo Jimenez Rezende, Phiala E. Shanahan"
"""
from __future__ import (
    absolute_import, print_function, division, annotations
)
import sys
from math import pi as PI

import torch
import torch.nn as nn

from typing import List
from fthmc.config import Param
from fthmc.utils.logger import Logger


TWO_PI = 2 * PI

logger = Logger()


def grab(var):
    return var.detach().cpu().numpy()

def put(s):
    sys.stdout.write(s)


def regularize(f):
    f_ = (f - PI) / TWO_PI
    return TWO_PI * (f_ - torch.floor(f_) - 0.5)


def torch_mod(x):
    return torch.remainder(x, TWO_PI)


def torch_wrap(x):
    return torch_mod(x + PI) - PI


def plaq_phase(f, mu=0, nu=1):
    f = torch.squeeze(f)
    if len(f.shape) == 4:
        return (f[:, mu]
                + torch.roll(f[:, nu], -1, mu + 1)
                - torch.roll(f[:, mu], -1, nu + 1)
                - f[:, nu])

    return (f[0, :]
            + torch.roll(f[1, :], -1, 0)
            - torch.roll(f[0, :], -1, 1)
            - f[1, :])

def u1_plaq(x: torch.Tensor, mu: int, nu: int):
    return (x[:, mu]
            + torch.roll(x[:, nu], -1, mu + 1)
            - torch.roll(x[:, mu], -1, nu + 1)
            - x[:, nu])


def topo_charge(x):
    phase = torch_wrap(u1_plaq(x, mu=0, nu=1))
    axes = tuple(range(1, len(phase.shape)))

    return torch.sum(phase, dim=axes) / TWO_PI


def compute_u1_plaq(links, mu=0, nu=1):
    """Compute U(1) plaqs in the (mu, nu) plane given `links` = arg(U)"""
    if len(links.shape) == 4:
        return (links[:, mu]
                - links[:, nu]
                - torch.roll(links[:, mu], -1, nu + 1)
                + torch.roll(links[:, nu], -1, mu + 1))
    return (links[mu, :]
            - links[nu, :]
            - torch.roll(links[mu, :], -1, nu + 1)
            + torch.roll(links[nu, :], -1, mu + 1))



def batch_plaqs(x: torch.Tensor, mu: int = 0, nu: int = 1):
    if len(x.shape) == 4:
        # x.shape = (batch, Nd, Nt, Nx)
        return (x[:, mu]
                - x[:, nu]
                - torch.roll(x[:, mu], -1, nu + 1)
                + torch.roll(x[:, nu], -1, mu + 1))

    return (x[mu, :],
            - x[nu, :]
            - torch.roll(x[mu, :], -1, nu + 1)
            + torch.roll(x[nu, :], -1, mu + 1))


def batch_charges(x: torch.Tensor = None, plaqs: torch.Tensor = None):
    if plaqs is None:
        if x is None:
            raise ValueError('Either `x` or `plaq` must be specified.')

        plaqs = torch_wrap(batch_plaqs(x, mu=0, nu=1))

    axes = tuple(range(1, len(plaqs.shape)))
    return torch.sum(plaqs, dim=axes) / TWO_PI


from dataclasses import dataclass

@dataclass
class LatticeMetrics:
    beta: float
    plaqs: torch.Tensor
    action: torch.Tensor
    charges: torch.Tensor



class BatchObservables:
    def __init__(self, beta: float = 1.):
        self.beta = beta

    def get_plaqs(self, x: torch.Tensor):
        d = x.shape[1]
        plaqs = 0
        for mu in range(d):
            for nu in range(mu + 1, d):
                p = batch_plaqs(x, mu, nu)
                plaqs += torch.cos(p)

        return plaqs

    def get_action(self, x: torch.Tensor = None, plaqs: torch.Tensor = None):
        if x is None:
            raise ValueError(f'Either `x` or `plaqs` must be specified.')

        d = x.shape[1]
        if plaqs is None:
            plaqs = self.get_plaqs(x)

        action = torch.sum(plaqs, dim=tuple(range(1, d+1)))
        return (-self.beta) * action

    def get_charges(self, x: torch.Tensor = None, plaqs: torch.Tensor = None):
        return batch_charges(x=x, plaqs=plaqs)

    def get_observables(self, x: torch.Tensor):
        plaqs = self.get_plaqs(x)
        action = self.get_action(plaqs=plaqs)
        charges = self.get_charges(plaqs=plaqs)

        return LatticeMetrics(self.beta, plaqs, action, charges)


class BatchAction:
    def __init__(self, beta):
        self.beta = beta

    @staticmethod
    def _u1_plaq(x: torch.Tensor, mu: int, nu: int):
        return (x[:, mu]                            # U_{mu}(x)
                + torch.roll(x[:, nu], -1, mu + 1)  # U_{nu}(x+mu)
                - torch.roll(x[:, mu], -1, nu + 1)  # -U_{mu}(x+nu)
                - x[:, nu])                         # -U_{nu}(x)

    def __call__(self, x: torch.Tensor):
        nd = x.shape[1]
        action_density = 0
        for mu in range(nd):
            for nu in range(mu + 1, nd):
                plaq = self._u1_plaq(x, mu, nu)
                action_density = action_density + torch.cos(plaq)

        action = torch.sum(action_density, dim=tuple(range(1, nd+1)))
        return (-self.beta) * action


#  Flow = List[torch.nn.Module]

def ft_flow(flow: nn.ModuleList, x: torch.Tensor):
    """Pass `x` through (forward) through each layer in `flow`."""
    #  if torch.cuda.is_available():
    #      f = f.cuda()
    for layer in flow:
        x, logdet = layer.forward(x)

    return x.detach()


def ft_flow_inv(flow: nn.ModuleList, x: torch.Tensor):
    """Pass"""
    #  if torch.cuda.is_available():
    #      f = f.cuda()

    for layer in reversed(flow):
        x, logdet = layer.reverse(x)

    return x.detach()


def ft_action(param: Param, flow: list[nn.Module], x: torch.Tensor):
    y = x.clone()
    logdet = 0.
    for layer in flow:
        y, logdet_ = layer(y)
        #  y, logdet_ = layer.forward(y)
        logdet += logdet_

    act_fn = BatchAction(param.beta)
    s = BatchAction(param.beta)(y) - logdet

    return s


def ft_force(
        param: Param,
        flow: list[nn.Module],
        field: torch.Tensor,
        create_graph=False
):
    """Field transformation force.

    Note: f is the field follows the transformed distribution (close to prior)
    """
    field.requires_grad_(True)
    s = ft_action(param, flow, field)
    ss = torch.sum(s)
    ff, = torch.autograd.grad(ss, field, create_graph=create_graph)
    field.requires_grad_(False)

    return ff



def plaq_phase(f, mu=0, nu=1):
    f = torch.squeeze(f)
    if len(f.shape) == 4:
        return (f[:, mu]
                + torch.roll(f[:, nu], -1, mu + 1)
                - torch.roll(f[:, mu], -1, nu + 1)
                - f[:, nu])

    return (f[0, :]
            + torch.roll(f[1, :], -1, 0)
            - torch.roll(f[0, :], -1, 1)
            - f[1, :])



def action(param: Param, x: torch.Tensor):
    return (-param.beta) * torch.sum(torch.cos(plaq_phase(x)))


def force(param: Param, x: torch.Tensor):
    x.requires_grad_(True)
    s = action(param, x)
    x.grad = None
    s.backward()
    dsdx = x.grad
    x.requires_grad_(False)
    return dsdx


def leapfrog(
        param: Param,
        x: torch.Tensor,
        p: torch.Tensor,
        verbose: bool = True
):
    dt = param.dt
    x_ = x + 0.5 * dt * p
    f = force(param, x_)
    p_ = p + (-dt) * f
    if verbose:
        plaq = action(param, x) / (-param.beta * param.volume)
        force_norm = torch.linalg.norm(f)
        logger.log(f'plaq(x): {plaq}, force_norm: {force_norm}')

    for _ in range(param.nstep - 1):
        x_ = x_ + dt * p_
        p_ = p_ + (-dt) * force(param, x_)

    x_ = x_ + 0.5 * dt * p_
    return (x_, p_)


def hmc(param, x, verbose=True):
    nb = x.shape[0]
    v = torch.randn_like(x)
    h0 = action(param, x) + 0.5 * torch.sum(v * v)
    x_, v_ = leapfrog(param, x, v, verbose=verbose)
    xr = regularize(x_)
    act = action(param, xr) + 0.5 * torch.sum(v_ * v_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act - h0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    newx = xr if acc else x

    return (dH, exp_mdH, acc, newx)
