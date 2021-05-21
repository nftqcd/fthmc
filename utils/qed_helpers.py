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

from utils.param import Param

TWO_PI = 2 * PI


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


def compute_u1_plaq(links, mu=0, nu=1):
    """Compute U(1) plaqs in the (mu, nu) plane given `links` = arg(U)"""
    if len(links.shape) == 4:
        return (links[:, mu]
                + torch.roll(links[:, nu], -1, mu + 1)
                - torch.roll(links[:, mu], -1, nu + 1)
                - links[:, nu])
    return (links[mu, :]
            + torch.roll(links[nu, :], -1, mu + 1)
            - torch.roll(links[mu, :], -1, nu + 1)
            - links[nu, :])


def plaq_phase(f, mu=0, nu=1):
    if len(f.shape) == 4:
        return (f[:, mu]
                + torch.roll(f[:, nu], -1, mu + 1)
                - torch.roll(f[:, mu], -1, nu + 1)
                - f[:, nu])

    return (f[0, :]
            + torch.roll(f[1, :], -1, 0)
            - torch.roll(f[0, :], -1, 1)
            - f[1, :])


def topo_charge(x):
    phase = torch_wrap(compute_u1_plaq(x, mu=0, nu=1))
    axes = tuple(range(1, len(phase.shape)))

    return torch.sum(phase, dim=axes) / TWO_PI


class BatchAction:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu + 1, Nd):
                plaq = compute_u1_plaq(cfgs, mu, nu)
                action_density = action_density + torch.cos(plaq)

        action = torch.sum(action_density, dim=tuple(range(1, Nd+1)))
        return - self.beta * action


from typing import List
Flow = List[torch.nn.Module]

def ft_flow(flow: Flow, x: torch.Tensor):
    """Pass `x` through (forward) through each layer in `flow`."""
    #  if torch.cuda.is_available():
    #      f = f.cuda()
    for layer in flow:
        x, logdet = layer.forward(x)

    return x.detach()


def ft_flow_inv(flow: list[nn.Module], x: torch.Tensor):
    """Pass"""
    #  if torch.cuda.is_available():
    #      f = f.cuda()

    for layer in reversed(flow):
        x, logdet = layer.reverse(x)

    return x.detach()


def ft_action(param, flow, f):
    y = f
    logdet = 0.
    for layer in flow:
        y, logdet_ = layer.forward(y)
        logdet += logdet_

    act_fn = BatchAction(param.beta)
    s = act_fn(y) - logdet

    return s


def ft_force(param, flow, field, create_graph=False):
    """Field transformation force.

    Note: f is the field follows the transformed distribution (close to prior)
    """
    field.requires_grad_(True)
    s = ft_action(param, flow, field)
    ss = torch.sum(s)
    ff, = torch.autograd.grad(ss, field, create_graph=create_graph)
    field.requires_grad_(False)

    return ff


def action(param, f):
    return (-param.beta)*torch.sum(torch.cos(plaq_phase(f)))


def force(param, f):
    f.requires_grad_(True)
    s = action(param, f)
    f.grad = None
    s.backward()
    ff = f.grad
    f.requires_grad_(False)
    return ff




def leapfrog(param, x, p, verbose=True):
    dt = param.dt
    x_ = x + 0.5 * dt * p
    f = force(param, x_)
    p_ = p + (-dt) * f
    if verbose:
        plaq = action(param, x) / (-param.beta * param.volume)
        force_norm = torch.linalg.norm(f)
        print(f'plaq(x): {plaq}, force_norm: {force_norm}')

    for i in range(param.nstep - 1):
        x_ = x_ + dt * p_
        p_ = p_ + (-dt) * force(param, x_)

    x_ = x_ + 0.5 * dt * p_
    return (x_, p_)


def hmc(param, x, verbose=True):
    p = torch.randn_like(x)
    act0 = action(param, x) + 0.5 * torch.sum(p * p)
    x_, p_ = leapfrog(param, x, p, verbose=verbose)
    xr = regularize(x_)
    act = action(param, xr) + 0.5 * torch.sum(p_ * p_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act - act0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    newx = xr if acc else x

    return (dH, exp_mdH, acc, newx)
