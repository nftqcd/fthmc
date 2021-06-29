"""
field_transformation.py

Implements `FieldTransformation` object for running HMC with trained flow.
"""
from __future__ import absolute_import, annotations, division, print_function

import time
from math import pi as PI
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

import fthmc.utils.plot_helpers as plotter
from fthmc.config import DTYPE, Param, TrainConfig, ftConfig
from fthmc.utils import qed_helpers as qed
from fthmc.utils.logger import Logger, in_notebook

#  import torch.autograd.functional as F  # noqa: F401


TWO_PI = 2. * PI

logger = Logger()


# pylint:disable=missing-function-docstring
def init_live_plots(
    xlabels: list,
    ylabels: list,
    colors: list = None,
    param: Param = None,
    config: TrainConfig = None,
    **kwargs,
):
    plots = {}
    if colors is None:
        colors = [f'C{i}' for i in range(10)]
    else:
        assert len(colors) == len(ylabels)
    for idx, (xlabel, ylabel) in enumerate(zip(xlabels, ylabels)):
        use_title = (idx == 0)
        plots[ylabel] = plotter.init_live_plot(param=param,
                                               config=config,
                                               color=colors[idx],
                                               xlabel=xlabel,
                                               ylabel=ylabel,
                                               use_title=use_title,
                                               **kwargs)

    return plots


def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph: bool = False):
    # xx, yy = x.detach().numpy(), y.detach().numpy()
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                      create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.

    return torch.stack(jac).reshape(y.shape + x.shape)


def write_summaries(
        metrics: dict,
        writer: SummaryWriter,
        step: int,
        pre: str = 'ftHMC'
):
    for key, val in metrics.items():
        if key == 'traj':
            continue
        #  if key == 'dt':
        if key in ['dt', 'acc']:
            val = torch.tensor(val, dtype=DTYPE)
        if len(val.shape) > 1:
            writer.add_histogram(f'{pre}/{key}', val, global_step=step)
        else:
            writer.add_scalar(f'{pre}/{key}', val.mean(), global_step=step)



torch._C._debug_only_display_vmap_fallback_warnings(True)


# pylint:disable=invalid-name, unnecessary-lambda, missing-function-docstring
# pylint:disable=no-member
class FieldTransformation(nn.Module):
    def __init__(
            self,
            flow: nn.ModuleList,
            config: TrainConfig,
            ftconfig: ftConfig,
    ):

        super().__init__()
        self.flow = flow        # layers of a `FlowModel`
        self.config = config    # Training config
        self.ftconfig = ftconfig
        self.tau = self.ftconfig.tau
        self.nstep = self.ftconfig.nstep
        self.dt = self.ftconfig.dt


        action_fn = qed.BatchAction(config.beta)
        self._action_fn = lambda x: action_fn(x)
        self._charge_fn = lambda x: qed.batch_charges(x=x).detach()
        self._action_sum = lambda x: self.action(x).sum(-1)
        self._action_sum_hmc = lambda x: self._action_fn(x).sum(-1)
        #  self._force_fn = lambda x: F.jacobian(self.action().sum(-1),
        #                                        inputs=x,
        #                                        vectorize=True,
        #                                        create_graph=True)
        #
        self._denom = (self.config.beta * self.config.volume)
        self._plaq_fn = lambda x: (
            ((-1.) * self._action_fn(x) / self._denom).detach()
        )

    def action(self, x: torch.Tensor):
        logdet = 0.
        for layer in self.flow:
            x, logJ = layer(x)
            logdet = logdet + logJ

        return self._action_fn(x) - logdet

    def flow_forward(self, x: torch.Tensor):
        xi = x
        logdet = 0.
        for layer in self.flow:
            xi, logdet_ = layer.forward(xi)
            logdet = logdet + logdet_

        return xi, logdet

    def flow_backward(self, x: torch.Tensor):
        logdet = 0.
        #  for layer in self.flow[::-1]:
        #  for layer in [self.flow][::-1]:
        for layer in reversed(self.flow):
            x, logdet_ = layer.reverse(x)
            logdet = logdet + logdet_

        return x, logdet

    def force(self, x: torch.Tensor):
        x = x.detach().requires_grad_()
        s = self.action(x)
        dsdx = torch.autograd.grad(s, x)[0]
        # free up GPU memory
        #  del x, s
        #  if torch.cuda.is_available():
        #      torch.cuda.empty_cache()

        return dsdx

    @staticmethod
    def wrap(x: torch.Tensor):
        return torch.remainder(x + PI, TWO_PI) - PI

    def calc_energy(self, x: torch.Tensor, v: torch.Tensor):
        #  return (self.action(x)
        #          + 0.5 * (v * v).reshape(x.shape[0], -1).sum(-1))
        return self.action(x) + (v * v).sum()

    def leapfrog(self, x: torch.Tensor, v: torch.Tensor):
        x_ = x + 0.5 * self.dt * v
        v_ = v + (-self.dt) * self.force(x_)

        for _ in range(self.nstep - 1):
            x_ = x_ + self.dt * v_
            v_ = v_ + (-self.dt) * self.force(x_)

        x = x + 0.5 * self.dt * v
        return x, v

    def hmc(self, x: torch.Tensor, step: int = None):
        if torch.cuda.is_available():
            x = x.cuda()

        t0 = time.time()
        metrics = {}
        if step is not None:
            metrics['traj'] = step

        x, _ = self.flow_backward(x)
        v = torch.randn_like(x)
        h0 = self.action(x) + 0.5 * (v * v).sum()

        x_, v_ = self.leapfrog(x, v)
        x_ = self.wrap(x_)
        h1 = self.action(x_) + 0.5 * (v_ * v_).sum()

        prob = torch.rand([], dtype=torch.float64)
        dh = h1 - h0
        exp_mdh = torch.exp(-dh)
        acc = prob < exp_mdh
        xnew = x_ if acc else x
        xout, _ = self.flow_forward(xnew)

        metrics.update(**{
            'dt': time.time() - t0,
            'acc': acc.detach(),
            'dh': dh.detach(),
        })

        return xout, metrics

    def initializer(self, rand: bool = True):
        nd = self.config.nd
        lat = self.config.lat
        if rand:
            x = torch.empty([nd, ] + lat).uniform_(0, TWO_PI)
        else:
            x = torch.zeros([nd, ] + lat)

        return x[None, :]


    def lattice_metrics(self, x: torch.Tensor, qold: torch.Tensor):
        q = self._charge_fn(x).detach()
        p = self._plaq_fn(x).detach()
        dq = torch.sqrt((q - qold) ** 2).detach()
        return {'plaq': p, 'q': q, 'dq': dq}

    def run(
            self,
            x: torch.Tensor = None,
            nprint: int = 1,
            nplot: int = 1,
            window: int = 10,
            num_trajs: int = None,
            writer: Optional[SummaryWriter] = None,
            plotdir: str = None,
            **kwargs,
    ):
        if x is None:
            x = self.initializer()

        if num_trajs is None:
            num_trajs = 1000

        #  for n in range(num_runs):
        if x is None:
            x = self.initializer()

        history = {}  # type: dict[str, list[torch.Tensor]]
        q = qed.batch_charges(x)
        p = (-1.) * self.action(x) / self._denom
        logger.print_metrics({'plaq': p, 'q': q})

        plots = {}
        if in_notebook():
            plots = init_live_plots(config=self.config,
                                    ylabels=['acc', 'dq', 'plaq'],
                                    xlabels=3 * ['trajectory'], **kwargs)

        # ------------------------------------------------------------
        # TODO: Create directories for `FieldTransformation` and
        # `save_live_plots` along with metrics, other plots to dirs
        # ------------------------------------------------------------
        for i in range(num_trajs):
            x, metrics_ = self.hmc(x, step=i)
            try:
                qold = history['q'][i-1]
            except KeyError:
                qold = q
            lmetrics = self.lattice_metrics(x, qold)
            metrics = {**metrics_, **lmetrics}

            for key, val in metrics.items():
                if writer is not None:
                    write_summaries(metrics, writer=writer,
                                    step=i, pre='ftHMC')
                try:
                    history[key].append(val)
                except KeyError:
                    history[key] = [val]
                #  history[key][i] = val

            if (i - 1) % nplot == 0 and in_notebook() and plots != {}:
                #  data = {
                #      k: history[k][:i, :] for k in ['dq', 'acc', 'plaq']
                #  }
                data = {
                    k: history[k] for k in ['dq', 'acc', 'plaq']
                }
                plotter.update_plots(plots, data, window=window)

            if i % nprint == 0:
                logger.print_metrics(metrics)  # , pre=pre)

            if plotdir is not None and in_notebook():
                plotter.save_live_plots(plots, outdir=plotdir)

        #  histories[n] = history
        plotter.plot_history(history,
                             config=self.config,
                             outdir=plotdir,
                             #  self.param,
                             therm_frac=0.0,
                             xlabel='Trajectory')

        return history
