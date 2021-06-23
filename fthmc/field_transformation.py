from __future__ import absolute_import, division, print_function, annotations
import time
from typing import Union
import torch
import torch.nn as nn
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
        self._charges_fn = lambda x: qed.batch_charges(x)

    def action(self, x: torch.Tensor):
        z, logdet = self.flow_forward(x)
        s = self._action_fn(z) - logdet

        #  return self._action_fn(z) - logdet
        return s.detach()

    def flow_forward(self, x: torch.Tensor):
        logdet = 0.
        for layer in self.flow:
            x, logdet_ = layer(x)
            logdet = logdet + logdet_

        return x, logdet

    def flow_backward(self, x: torch.Tensor):
        logdet = 0.
        #  for layer in self.flow[::-1]:
        #  for layer in [self.flow][::-1]:
        for layer in reversed(self.flow):
            x, logdet_ = layer.reverse(x)
            logdet = logdet - logdet_

        return x, logdet

    def force(self, x: torch.Tensor, create_graph: bool = False):
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
        dsdx, = torch.autograd.grad(s, x, create_graph=create_graph,
                                    grad_outputs=torch.ones(x.shape[0]))
        #  x.requires_grad_(False)
        x.detach_()
        #  dsdx, = torch.autograd.grad(s.sum(), x, create_graph=create_graph)
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
        v = v + (-dt) * self.force(x)

        for _ in range(self.param.nstep - 1):
            x = x + dt * v
            v = v + (-dt) * self.force(x)

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
        if x is None:
            x = self.param.initializer()

        #x0 = x.clone()
        x0 = x.clone()
        nb = x.shape[0]
        v = torch.randn_like(x)
        h0 = self.calc_energy(x, v)

        x, v = self.leapfrog(x, v)
        x = self.wrap(x)

        h1 = self.calc_energy(x, v)

        dh = h1 - h0
        exp_mdh = torch.exp(-dh)
        acc = (torch.rand_like(exp_mdh) < exp_mdh)
        x_ = torch.cat([x[acc], x0[torch.logical_not(acc)]])

        return x_, acc, dh, exp_mdh


    def run(
            self,
            x: torch.Tensor = None,
            nprint: int = 1,
            nplot: int = 1,
            dpi: int = 120,
            figsize: tuple = None,
            window: int = 0,
    ):
        if x is None:
            x = self.param.initializer()
            x = x[None, :]

        nb = x.shape[0]
        runs_history = {}
        beta = self.param.beta
        volume = self.param.volume
        if figsize is None:
            figsize = (8, 2.75)

        plots_acc = plotter.init_live_plot(dpi=dpi,
                                           figsize=figsize,
                                           ylabel='acc',
                                           color='#F92672',
                                           param=self.param,
                                           xlabel='trajectory',
                                           config=self.config)
        plots_dqsq = plotter.init_live_plot(dpi=dpi,
                                            figsize=figsize,
                                            color='#00CCff',
                                            ylabel='dqsq',
                                            xlabel='trajectory')
        plots_plaq = plotter.init_live_plot(figsize=figsize,
                                            dpi=dpi,
                                            color='#ffff00',
                                            ylabel='plaq',
                                            xlabel='trajectory')

        plots = {'plaq': plots_plaq, 'dqsq': plots_dqsq, 'acc': plots_acc}

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
                x, acc, dh, exp_mdh = self.build_trajectory(x)
                #  qold = self._charges_fn(x)
                #q0 = qed.(x[None, :])

                #x = self.
                #  qold = history['q'][-1]
                qold = qarr[i-1]
                qnew = qed.batch_charges(x)
                #  qnew = self._charges_fn(x)
                #q1 = qed.batch_charges(x[None, :])

                dqsq = (qnew - qold) ** 2
                #  dqsq_avg = dqsq.mean()
                #  dqsq_avg = (dqsq[-window:])
                #logp = (-1.) * self._action_fn(x)
                #  logp = (-1.) * self.action(x)
                plaq = (-1.) * self.action(x) / (beta * volume)
                #  plaq_no_grad = plaq.detach()

                qarr[i] = qnew
                metrics = {
                    'traj': n * self.param.ntraj + i + 1,
                    'acc': acc,
                    'dh': dh,
                    'exp_mdh': exp_mdh,
                    'dqsq': dqsq,
                    #  'q': int(qnew),
                    'plaq': plaq,
                }

                history['dt'].append(time.time() - t_)
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
                    plotter.update_plots(plots, data)  # , window=10)


                if i % nprint == 0:
                    logger.print_metrics(history, skip=['q'], pre=['(now)'],
                                         window=0)
                    logger.print_metrics(history, skip=['q'], pre=['(avg)'],
                                         window=10)
                #logger.print_metrics(metrics, skip=['q'], pre=['(avg)'],
                #                     window=min(i, 20))

            runs_history[n] = history

        return runs_history
