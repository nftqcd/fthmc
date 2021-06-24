from __future__ import absolute_import, division, print_function, annotations
import time
from typing import Union
import torch
import torch.nn as nn
import torch.autograd.functional as F
import numpy as np

from fthmc.utils import qed_helpers as qed
from fthmc.config import TrainConfig, Param, grab
import fthmc.utils.plot_helpers as plotter
#  from fthmc.utils.plot_helpers import init_live_plot, update_plots


from math import pi as PI

TWO_PI = 2. * PI

from fthmc.utils.logger import Logger

logger = Logger()


def grab(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return float(x)


def init_live_plots(
    param: Param,
    xlabels: list,
    ylabels: list,
    #  dpi: int = 120,
    #  config: TrainConfig = None,
    #  figsize: tuple = None,
    colors: list = None,
    config: TrainConfig = None,
    **kwargs,
):
    plots = {}
    if colors is None:
        colors = [f'C{i}' for i in range(10)]
    else:
        assert len(colors) == len(ylabels)
    for idx, (xlabel, ylabel) in enumerate(zip(xlabels, ylabels)):
        plots[ylabel] = plotter.init_live_plot(param=param,
                                               config=config,
                                               color=colors[idx],
                                               xlabel=xlabel,
                                               ylabel=ylabel,
                                               **kwargs)

    return plots


def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph: bool = False):
    # xx, yy = x.detach().numpy(), y.detach().numpy()
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.

    return torch.stack(jac).reshape(y.shape + x.shape)


Flow = Union[nn.ModuleList, list[nn.Module]]

class FieldTransformation(nn.Module):
    def __init__(
            self,
            flow: Flow,
            param: Param,
            config: TrainConfig = None
    ):
        super().__init__()
        self.param = param
        self.flow = flow
        self.config = config
        action_fn = qed.BatchAction(param.beta)
        self._action_fn = lambda x: action_fn(x)
        self._action_denom = (self.param.beta * self.param.volume)
        self._charges_fn = lambda x: qed.batch_charges(x)
        action_sum = lambda x: self.action(x).sum(-1)
        self._dsdx_fn = lambda x: F.jacobian(action_sum, x)

    def action(self, x: torch.Tensor):
        z, logdet = self.flow_forward(x)
        s = self._action_fn(z) - logdet

        #  return self._action_fn(z) - logdet
        return s.detach()

    def flow_forward(self, x: torch.Tensor):
        xi = x
        logdet = 0.
        for layer in self.flow:
            xi, logdet_ = layer(xi)
            logdet = logdet + logdet_

        return xi, logdet

    def flow_backward(self, x: torch.Tensor):
        logdet = 0.
        #  for layer in self.flow[::-1]:
        #  for layer in [self.flow][::-1]:
        for layer in reversed(self.flow):
            x, logdet_ = layer.reverse(x)
            logdet = logdet - logdet_

        return x, logdet

    def force(self, x: torch.Tensor):
        dsdx = self._dsdx_fn(x)
        x.detach_()
        return dxdx

    def force1(self, x: torch.Tensor, **kwargs):
        #  s = torch.tensor(0., requires_grad=True)
        x.requires_grad_(True)
        s = 0.
        logdet = 0.
        for layer in self.flow:
            x, logdet_ = layer(x)
            logdet = logdet + logdet_

        #  z, logdet = self.flow_forward(x)
        s = self._action_fn(x) - logdet
        #  s = self.action(x)
        dx = torch.ones(x.shape[0])
        dsdx, = torch.autograd.grad(s, x, grad_outputs=dx, **kwargs)
        #  x.requires_grad_(False)
        x.detach_()
        #  dsdx, = torch.autograd.grad(s.sum(), x, create_graph=create_graph)
        #  x.requires_grad_(False)

        return dsdx

    def _force(self, x: torch.Tensor, **kwargs):
        x.requires_grad_(True)
        s_ = self.action(x)
        s = s_.sum()
        dsdx, = torch.autograd.grad(s, x, **kwargs)
        x.detach_()
        #  x.requires_grad_(False)

        return dsdx


    @staticmethod
    def wrap(x: torch.Tensor):
        mod = lambda x: torch.remainder(x, TWO_PI)
        return mod(x + PI) - PI

    def wrap1(x: torch.Tensor):
        x = (x - PI) / TWO_PI

        return TWO_PI * (x - x.floor() - 0.5)

    def calc_energy(self, x: torch.Tensor, v: torch.Tensor):
        if len(x.shape) == 3:
            # [Nd, Nt, Nx]
            x = x[None, :]

        nb = x.shape[0]

        return self.action(x) + 0.5 * (v * v).reshape(nb, -1).sum(-1)

    def leapfrog(self, x: torch.Tensor, v: torch.Tensor):
        dt = self.param.dt

        x = x + 0.5 * dt * v
        v = v + (-dt) * self.force(x, retain_graph=True)

        for _ in range(self.param.nstep - 1):
            x = x + dt * v
            v = v + (-dt) * self.force(x, retain_graph=True)

        x = x + 0.5 * dt * v
        return x.detach(), v

    def forward1(self, x: torch.Tensor):
        v0 = torch.randn_like(x)

        v0_norm = (v0 * v0).sum()
        logdet = 0.
        for layer in self.flow:
            x, logdet_ = layer(x)
            logdet = logdet + logdet_

        action = self._action_fn(x) - logdet

        plaq0 = self.action(x) + 0.5 * v0_norm

        x, v1 = self.leapfrog(x, v)
        xr = self.wrap(x)

        v1_norm = (v1 * v1).sum()
        plaq1 = self.action(x) + 0.5 * v1_norm

        prob = torch.rand(x.shape[0])
        dH = plaq1 - plaq0
        exp_mdH = torch.exp(-dH)
        acc = (prob < exp_mdH).float()
        x_ = xr if acc else x

        prob = torch.exp(torch.minimum(dH, torch.zeros_like(dH)))
        acc = torch.rand(prob.shape) < prob
        pass

    def build_trajectory(self, x: torch.Tensor = None):
        t0 = time.time()
        if x is None:
            x = self.param.initializer()

        nb = x.shape[0]

        x0 = x.clone()

        q0 = self._charges_fn(x0)

        v0 = torch.randn_like(x0)
        h0 = self.calc_energy(x0, v0)

        x1_, v1 = self.leapfrog(x0, v0)
        x1 = self.wrap(x1_)

        h1 = self.calc_energy(x1, v1)

        dh = h1 - h0
        exp_mdh = torch.exp(-dh)
        acc = (torch.rand_like(exp_mdh) < exp_mdh).int()
        x_ = acc * x1 + (1 - acc) * x0
        #  x_ = torch.zeros_like(x0)
        # x_ = torch.stack([x1[acc], x0[torch.logical_not(acc)]])

        q1 = self._charges_fn(x_)
        p1 = (-1.) * self.action(x_) / self._action_denom
        dqsq = (q1 - q0) ** 2
        #  xout, _ = self.flow_backward(x)
        xout, _ = self.flow_forward(x_)
        metrics = {
            'dt': time.time() - t0,
            'acc': acc, 'dqsq': dqsq,
            'q': q1, 'plaq': p1,
            'dh': dh, 'exp_mdh': exp_mdh,
        }
        #  return xout, acc, dh, exp_mdh
        return xout, metrics


    def run(
            self,
            x: torch.Tensor = None,
            nprint: int = 1,
            nplot: int = 1,
            window: int = 0,
            **kwargs,
            #  dpi: int = 120,
            #  figsize: tuple = None,
    ):
        if x is None:
            x = self.param.initializer()
            x = x[None, :]

        nb = x.shape[0]
        runs_history = {}
        beta = self.param.beta
        volume = self.param.volume
        #  if figsize is None:
        #      figsize = (8, 2.75)

        ylabels = ['acc', 'dqsq', 'plaq']
        xlabels = len(ylabels) * ['trajectory']
        plots = init_live_plots(self.param,
                                xlabels=xlabels,
                                ylabels=ylabels, **kwargs)

        for n in range(self.param.nrun):
            t0 = time.time()
            xarr = []
            history = {}
            qarr = torch.zeros((self.param.ntraj, x.shape[0]),
                               requires_grad=False)
            if x is None:
                x = self.param.initializer()[None, :]

            p = (-1.) * self.action(x) / (beta * volume)
            q = qed.batch_charges(x)
            logger.print_metrics({'plaq': p, 'q': q})
            xarr = []
            #  zeros = torch.zeros((self.param.ntraj, nb), requires_grad=False)
            #  ones = torch.ones((self.param.ntraj, nb), requires_grad=False)
            zeros = torch.zeros(nb, requires_grad=False)
            ones = torch.ones(nb, requires_grad=False)
            #  history = {
            #      'traj': torch.zeros((self.param.ntraj, nb), requires_grad=False),
            #      'dt': torch.zeros((self.param.ntraj, nb), requires_grad=False),
            #      'acc': torch.ones((self.param.ntraj, nb), requires_grad=False),
            #      'dh': torch.zeros((self.param.ntraj, nb), requires_grad=False),
            #      'exp_mdh': torch.ones((self.param.ntraj, nb), requires_grad=False),
            #      'dqsq': torch.zeros((self.param.ntraj, nb), requires_grad=False),
            #      'plaq': torch.ones((self.param.ntraj, nb), requires_grad=False),
            #  }
            #  history = {
            #      'traj': zeros.clone(),
            #      'dt': zeros.clone(),
            #      'acc': ones.clone(),
            #      'dh': zeros.clone(),
            #      'exp_mdh': ones.clone(),
            #      'dqsq': zeros.clone(),
            #      'plaq': ones.clone(),
            #  }
            history = {
                'traj': [0],
                'dt': [0.],
                'acc': [grab(ones.clone())],
                'dH': [grab(zeros.clone())],
                'exp_mdh': [grab(ones.clone())],
                'dqsq': [grab(zeros.clone())],
                'q': [grab(q)],
                'plaq': [grab(p)],
            }

            for i in range(1, self.param.ntraj):
                t_ = time.time()
                #  x, acc, dh, exp_mdh = self.build_trajectory(x)
                x, metrics = self.build_trajectory(x)
                #  qold = self._charges_fn(x)
                #q0 = qed.(x[None, :])

                #x = self.
                #  qold = history['q'][-1]
                #  qold = qarr[-1]
                #  qnew = qed.batch_charges(x)
                #  qnew = self._charges_fn(x)
                #q1 = qed.batch_charges(x[None, :])

                #  dqsq = (qnew - qold) ** 2
                #  dqsq_avg = dqsq.mean()
                #  dqsq_avg = (dqsq[-window:])
                #logp = (-1.) * self._action_fn(x)
                #  logp = (-1.) * self.action(x)
                #  plaq = (-1.) * self.action(x) / (beta * volume)
                #  plaq_no_grad = plaq.detach()

                qarr[i] = metrics['q']
                #  metrics['traj'] = i
                #  metrics['dt'] = time.time() - t0
                #  metrics = {
                #      #  'traj': n * self.param.ntraj + i + 1,
                #      'traj': i,
                #      'acc': acc,
                #      'dh': dh,
                #      'exp_mdh': exp_mdh,
                #      'dqsq': dqsq,
                #      #  'q': int(qnew),
                #      'plaq': plaq,
                #  }
#
                #  history['dt'].append(time.time() - t_)
                for key, val in metrics.items():
                    #  history[key][i] = val
                    try:
                        history[key].append(grab(val))
                    except KeyError:
                        history[key] = [grab(val)]

                if i % nplot == 0:
                    data = {
                        #  'q': grab(qarr[:, 0]),
                        'dqsq': history['dqsq'],
                        #  'dqsq': dqsq_avg.detach().numpy(),
                        'acc': history['acc'],
                        'plaq': history['plaq'],
                    }
                    plotter.update_plots(plots, data, window=10)


                if i % nprint == 0:
                    pre = ['(now)', f'traj={i}']
                    logger.print_metrics(history, skip=['q'], pre=pre,
                                         window=0)
                    pre = ['(avg)', f'traj={i}']
                    logger.print_metrics(history, skip=['q'], pre=['(avg)'],
                                         window=10)
                #logger.print_metrics(metrics, skip=['q'], pre=['(avg)'],
                #                     window=min(i, 20))

            runs_history[n] = history

        return runs_history
