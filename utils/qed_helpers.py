"""
utils/qed_helpers.py

Contains various helper functions for training the normalizing flow model on a
two-dimensional U(1) (QED) theory.
"""
import sys
import torch

from utils.param import Param
import utils.field_transformation as ft
from math import pi as PI

TWO_PI = 2 * PI


def ft_flow(flow, f):
    for layer in flow:
        f, logdet = layer.forward(f)

    return f.detach()


def ft_flow_inv(flow, f):
    for layer in reversed(flow):
        f, logdet = layer.reverse(f)

    return f.detach()

def ft_action(param, flow, field):
    y = field
    logdet = 0.
    for layer in flow:
        y, logdet_ = layer.forward(y)
        logdet += logdet_

    s = action(param, y) - logdet
    #  action = ft.U1GaugeAction(param.beta)

    return s

def ft_force(param, flow, field, create_graph=False):
    # f is the field follows the transformed distribution (close to prior)
    field.requires_grad_(True)
    s = ft_action(param, flow, field)
    #  s = ft_action(param, flow, field)
    ss = torch.sum(s)
    ff, = torch.autograd.grad(ss, field, create_graph=create_graph)
    field.requires_grad_(False)

    return ff


def put(s):
    sys.stdout.write(s)


def plaq_phase(f):
    return (f[0, :]
            - f[1, :]
            - torch.roll(f[0, :], shifts=-1, dims=1)
            + torch.roll(f[1, :], shifts=-1, dims=0))


def regularize(f):
    f_ = (f - PI) / TWO_PI
    return TWO_PI * (f_ - torch.floor(f_) - 0.5)


def topo_charge(f):
    return torch.floor(0.1 + torch.sum(regularize(plaq_phase(f))) / TWO_PI)


def action(param, f):
    return (-param.beta) * torch.sum(torch.cos(plaq_phase(f))) / TWO_PI


def force(param, f):
    f.requires_grad_(True)
    s = action(param, f)
    f.grad = None
    s.backward()
    df = f.grad
    f.requires_grad_(False)

    return df


def leapfrog(param, x, p, verbose=True):
    dt = param.dt
    x_ = x + 0.5 * dt * p
    f = force(param, x_)
    p_ = p + (-dt) * f

    if verbose:
        print(f'plaq(x): {action(param, x) / (-param.beta * param.volume):<.4g}, '
              f'force.norm {torch.linalg.norm(f):>.4g}')

    for i in range(param.nstep - 1):
        x_ = x_ + dt * p
        p_ = p_ + (-dt) * force(param, x_)

    x_ = x_ + 0.5 * dt * p_

    return (x_, p_)


def hmc(param, x, verbose=True):
    p = torch.rand_like(x)
    act0 = action(param, x) + 0.5 * torch.sum(p*p)
    x_, p_ = leapfrog(param, x, p, verbose)
    xr = regularize(x_)
    act = action(param, xr) + 0.5 * torch.sum(p_ * p_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act - act0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    newx = xr if acc else x

    return (dH, exp_mdH, acc, newx)
