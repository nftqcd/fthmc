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
        #y = x

        #logdet = 0.
        #for layer in self.flow:
        #    y, logdet_ = layer(y)
        #    logdet = logdet + logdet_
        #
        #return self._action_fn(y) - logdet
        z, logdet = self.flow_forward(x)

        return self._action_fn(z) - logdet

    def flow_forward(self, x: torch.Tensor):
        logdet = 0.
        for layer in self.flow:
            x, logdet_ = layer(x)
            logdet = logdet + logdet_

        return x, logdet

    def flow_backward(self, x: torch.Tensor):
        logdet = 0.
        #  for layer in self.flow[::-1]:
        for layer in [self.flow][::-1]:
            x, logdet_ = layer.reverse(x)
            logdet = logdet - logdet_

        return x, logdet

    def force(self, x: torch.Tensor, create_graph: bool = False):
        x.requires_grad_(True)
        s = self.action(x)
        dsdx, = torch.autograd.grad(s.sum(), x, create_graph=create_graph)
        #  x.requires_grad_(False)

        return dsdx

    @staticmethod
    def wrap(x: torch.Tensor):
        x = (x - PI) / TWO_PI

        return TWO_PI * (x - x.floor() - 0.5)

    def calc_energy(self, x: torch.Tensor, v: torch.Tensor):
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
        return x, v

    def build_trajectory(self, x: torch.Tensor = None):
        if x is None:
            x = self.param.initializer()[None, :]

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
        acc = (torch.rand_like(exp_mdh) < exp_mdh).float()

        x_ = x.reshape(nb, -1)
        x_ = x.reshape(nb, -1)
        x_ = acc * x + (1 - acc) * x0

        xout, _  = self.flow_forward(x_)
        return self.wrap(xout), acc, dh, exp_mdh

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
            qarr = torch.zeros((self.param.ntraj, x.shape[0]))
            for i in range(self.param.ntraj):
                t_ = time.time()
                q0 = self._charges_fn(x)
                #q0 = qed.(x[None, :])

                x, acc, dH, exp_mdH = self.build_trajectory(x)
                #x = self.
                q1 = self._charges_fn(x)
                #q1 = qed.batch_charges(x[None, :])

                dqsq = (q1 - q0) ** 2
                #logp = (-1.) * self._action_fn(x)
                logp = (-1.) * self.action(x)
                plaq = logp / (beta * volume)
                plaq_no_grad = plaq.detach()

                qarr[i] = q1
                metrics = {
                    'traj': i,
                    'dt': time.time() - t_,
                    'acc': acc,
                    'dH': dH,
                    'exp_mdH': exp_mdH,
                    'dqsq': dqsq,
                    'q': q1,
                    'plaq': plaq_no_grad,
                }

                for key, val in metrics.items():
                    try:
                        history[key].append(grab(val))
                    except KeyError:
                        history[key] = [grab(val)]

                if i % nplot == 0:
                    #  acc_ = torch.Tensor([list(i) for i in
                    #                       history['acc']).mean(-1)
                    #acc_ = torch.Tensor(history).mean(-1)
                    dqsq_avg = np.array(history['dqsq']).mean(axis=-1)

                    #  acc_ = history['acc'][-1].mean()
                    #  dqsq_ = history['dqsq'][-1].mean()
                    #  plaq_ = history['plaq'][-1].mean()
                    #  q = qarr[:, 0]
                    #  plaq_ = torch.Tensor(history['plaq']).mean(-1)
                    data = {
                        #  'q': grab(qarr[:, 0]),
                        'dqsq': dqsq_avg,
                        'acc': history['acc'],
                        'plaq': history['plaq'],
                    }
                    plotter.update_plots(plots, data, window=5)


                if i % nprint == 0:
                    logger.print_metrics(metrics, skip=['q'], pre=['(now)'],
                                         window=0)
                    logger.print_metrics(metrics, skip=['q'], pre=['(avg)'],
                                         window=10)
                #logger.print_metrics(metrics, skip=['q'], pre=['(avg)'],
                #                     window=min(i, 20))

            runs_history[n] = history

        return runs_history
