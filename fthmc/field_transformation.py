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
        self._denom = (self.param.beta * self.param.volume)
        self._charges_fn = lambda x: qed.batch_charges(x).detach()
        self._plaq_fn = lambda x: (
            ((-1.) * self.action(x) / self._denom).detach()
        )
        action_sum = lambda x: self.action(x).sum(-1)
        self._dsdx_fn = lambda x: F.jacobian(action_sum, x)

    def action(self, x: torch.Tensor):
        z, logdet = self.flow_forward(x)
        s = self._action_fn(z) - logdet
        return s.detach()

    def flow_forward(self, x: torch.Tensor):
        xi = x
        logdet = 0.
        for layer in self.flow:
            xi, logdet_ = layer(xi)
            logdet = logdet - logdet_

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
        return dsdx

    def force_old(self, x: torch.Tensor, **kwargs):
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

    @staticmethod
    def wrap_old(x: torch.Tensor):
        x = (x - PI) / TWO_PI

        return TWO_PI * (x - x.floor() - 0.5)


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

    def calc_energy(self, x: torch.Tensor, v: torch.Tensor):
        nb = x.shape[0]

        return self.action(x) + 0.5 * (v * v).reshape(nb, -1).sum(-1)

    def leapfrog(self, x: torch.Tensor, v: torch.Tensor):
        dt = self.param.dt

        x = x + 0.5 * dt * v
        v = v + (-dt) * self.force(x)  # , retain_graph=True)

        for _ in range(self.param.nstep - 1):
            x = x + dt * v
            v = v + (-dt) * self.force(x)  # , retain_graph=True)

        x = x + 0.5 * dt * v
        return x, v

    def build_trajectory(self, x: torch.Tensor = None, step: int = None):
        t0 = time.time()
        if x is None:
            x = self.param.initializer()

        metrics = {}
        if step is not None:
            metrics = {'traj': step}

        #  xshape = x.shape
        #  nb = x.shape[0]

        x0 = x.clone()

        q0 = self._charges_fn(x0)

        v0 = torch.randn_like(x0)
        h0 = self.calc_energy(x0, v0)

        x1_, v1 = self.leapfrog(x0, v0)
        x1 = self.wrap(x1_)

        h1 = self.calc_energy(x1, v1)

        dh = h1 - h0
        exp_mdh = torch.exp(-dh)
        acc = (torch.rand_like(exp_mdh) < exp_mdh)
        m = acc[:, None].to(torch.uint8)
        x_ = m * x1.flatten(start_dim=1) + (1 - m) * x0.flatten(start_dim=1)
        x_ = x_.reshape(x0.shape)
        #  x_ = torch.zeros(x0.shape)
        #  x_ = acc * x1 + (1 - acc) * x0
        #  xout, _ = self.flow_backward(x)
        xout, _ = self.flow_forward(x_)

        q1 = self._charges_fn(xout)
        p1 = self._plaq_fn(xout)
        dq = torch.sqrt((q1 - q0) ** 2)
        metrics.update(**{
            'dt': time.time() - t0,
            'acc': acc.float().detach(),
            'dh': dh,
            'exp_mdh': exp_mdh,
            'q': q1.to(torch.int).detach(),
            'dq': dq.detach(),
            'plaq': p1.detach(),
        })

        return xout.detach(), metrics


    def run(
            self,
            x: torch.Tensor = None,
            nprint: int = 1,
            nplot: int = 1,
            **kwargs,
            #  dpi: int = 120,
            #  figsize: tuple = None,
    ):
        if x is None:
            x = self.param.initializer()
            #  x = x[None, :]
            #  x = x[None, :]

        #  nb = x.shape[0]
        runs_history = {}

        ylabels = ['acc', 'dq', 'plaq']
        xlabels = len(ylabels) * ['trajectory']
        plots = init_live_plots(self.param, xlabels=xlabels,
                                ylabels=ylabels, **kwargs)

        for n in range(self.param.nrun):
            t0 = time.time()
            #  xarr = []

            #  q = self._charges_fn(x)
            #  p = self._plaq_fn(x)

            #  zero = torch.tensor(0., dtype=torch.float)
            #  onef = torch.tensor(1., dtype=torch.float)
            #  onei = torch.tensor(1, dtype=torch.int)

            #  history = {
            #      'dt': [0.],
            #      'traj': [0],
            #      'acc': [],
            #      'dh': [zero],
            #      'exp_mdh': [onef],
            #      'q': [q],
            #      'dqsq': [0],
            #      'plaq': [p],
            #  }
            history = {}
            scalar_zeros = torch.zeros(self.param.ntraj, requires_grad=False)
            batch_zeros = torch.zeros((self.param.ntraj, x.shape[0]),
                                      requires_grad=False)
            qarr = batch_zeros.clone()
            #  qarr = torch.zeros((self.param.ntraj, x.shape[0]),
            #                     requires_grad=False)
            if x is None:
                x = self.param.initializer()

            #  p = (-1.) * self.action(x) / (beta * volume).det
            #  q = qed.batch_charges(x)
            #  logger.print_metrics({'plaq': p, 'q': q})
            #  xarr = []

            for i in range(1, self.param.ntraj):
                t_ = time.time()
                x, metrics = self.build_trajectory(x, step=i)

                #  if i == 1:
                #      history['traj'] = [0]
                #  else:
                #      history['traj'].append(i)
                qarr[i] = metrics['q']
                for key, val in metrics.items():
                    if isinstance(val, (list, np.ndarray, torch.Tensor)):
                        if len(val) == 1:
                            val = val[0]
                    try:
                        history[key].append(val)
                    except KeyError:
                        history[key] = [val]

                if (i - 1) % nplot == 0:
                    data = {
                        'dq': history['dq'],
                        'acc': history['acc'],
                        'plaq': history['plaq'],
                    }
                    plotter.update_plots(plots, data, window=5)  # , window=10)


                if i % nprint == 0:
                    #  pre = ['(now)', f'traj={i}']
                    logger.print_metrics(metrics)  # , pre=pre)
                    #  logger.print_metrics(history, skip=['q'], pre=pre,
                    #                       window=0)
                    #  pre = ['(avg)', f'traj={i}']
                    #  logger.print_metrics(history, skip=['q'], pre=['(avg)'],
                    #                       window=10)
                #logger.print_metrics(metrics, skip=['q'], pre=['(avg)'],
                #                     window=min(i, 20))

            hist = {}
            for key, val in history.items():
                if key in ['traj', 'dt']:
                    vt = torch.tensor(val, dtype=torch.float)
                else:
                    vt = torch.Tensor(torch.stack(val).to(torch.float))

                hist[key] = vt

            plotter.plot_history(hist,
                                 self.param,
                                 therm_frac=0.0,
                                 xlabel='Trajectory')

        return runs_history
