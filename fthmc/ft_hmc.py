"""
field_transformation.py

Implements `FieldTransformation` object for running HMC with trained flow.
"""
from __future__ import absolute_import, annotations, division, print_function

import os
import time
from math import pi as PI
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

import fthmc.utils.io as io
import fthmc.utils.plot_helpers as plotter
from fthmc.config import DTYPE, Param, TrainConfig, lfConfig
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
    """Initialize live plots for ftHMC."""
    plots = {}
    if colors is None:
        colors = [f'C{i}' for i in range(10)]
    else:
        assert len(colors) == len(ylabels)
    for idx, (xlabel, ylabel) in enumerate(zip(xlabels, ylabels)):
        use_title = (idx == 0)
        plots[ylabel] = plotter.init_live_plot(param=param, config=config,
                                               xlabel=xlabel, ylabel=ylabel,
                                               color=colors[idx],
                                               use_title=use_title, **kwargs)

    return plots


def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph: bool = False):
    """Generic function for computing the Jacobian, dy/dx."""
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
    """Write summaries of items in `metrics` using `writer`."""
    for key, val in metrics.items():
        if key == 'traj':
            continue
        #  if key == 'dt':
        # if key in ['dt', 'acc']:
        if isinstance(val, float):
            writer.add_scalar(f'{pre}/{key}', val, global_step=step)
        elif isinstance(val, torch.Tensor):
            val = val.detach().type(torch.get_default_dtype())
            if len(val.shape) > 1:
                writer.add_histogram(f'{pre}/{key}', val, global_step=step)
        else:
            val = torch.tensor(val, dtype=DTYPE)
            writer.add_scalar(f'{pre}/{key}', val.mean(), global_step=step)
        # if isinstance(val, torch.Tensor):
        #     val = val.detach().type(torch.get_default_dtype())
        # if isinstance(val, float):
        #     writer.add_scalar(f'{pre}/{key}', val, global_step=step)
        # elif len(val.shape) > 1:
        #     writer.add_histogram(f'{pre}/{key}', val, global_step=step)
        # else:
        #     writer.add_scalar(f'{pre}/{key}', val.mean(), global_step=step)



torch._C._debug_only_display_vmap_fallback_warnings(True)


# pylint:disable=no-member
# pylint:disable=invalid-name, unnecessary-lambda, missing-function-docstring
class FieldTransformation(nn.Module):
    def __init__(
            self,
            flow: nn.ModuleList,
            config: TrainConfig,
            lfconfig: lfConfig,
    ):
        super().__init__()
        self.flow = flow                    # layers of a `FlowModel`
        self.config = config                # Training config
        self.lfconfig = lfconfig            # lfConfig object
        self.dt = self.lfconfig.dt          # step size
        self.tau = self.lfconfig.tau        # trajectory length
        self.nstep = self.lfconfig.nstep    # number of leapfrog steps
        self._denom = (self.config.beta * self.config.volume)


        action_fn = qed.BatchAction(config.beta)
        self._action_fn = lambda x: action_fn(x)
        self._charge_fn = lambda x: qed.batch_charges(x=x).detach()
        self._action_sum = lambda x: self.action(x).sum(-1)
        self._action_sum_hmc = lambda x: self._action_fn(x).sum(-1)
        self._plaq_fn = lambda x: (
            ((-1.) * self._action_fn(x) / self._denom).detach()
        )

    def action(self, x: torch.Tensor):
        logdet = 0.
        for layer in self.flow:
            x, logJ = layer.forward(x)
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
        return self.action(x) + (v * v).sum()

    def leapfrog(self, x: torch.Tensor, v: torch.Tensor):
        x_ = x + 0.5 * self.dt * v
        v_ = v + (-self.dt) * self.force(x_)
        for _ in range(self.nstep - 1):
            x_ = x_ + self.dt * v_
            v_ = v_ + (-self.dt) * self.force(x_)

        x = x + 0.5 * self.dt * v
        return x, v

    def hmc(
            self,
            x: torch.Tensor,
            step: int = None
    ):
        if torch.cuda.is_available():
            x = x.cuda()

        t0 = time.time()
        metrics = {}
        if step is not None:
            metrics['traj'] = step

        #  x, _ = self.flow_backward(x)
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
        #  xout, _ = self.flow_forward(xnew)

        metrics.update(**{
            'dt': time.time() - t0,
            'acc': acc.detach(),
            'dh': dh.detach(),
        })

        return xnew, metrics

    def _batch_hmc(
            self,
            x: torch.Tensor,
            step: int = None,
    ):
        t0 = time.time()
        metrics = {}  # type: dict[str, Union[torch.Tensor, float]]
        if step is not None:
            metrics['traj'] = step

        #  x0, _ = self.flow_backward(x)
        v = torch.randn_like(x)
        h = self.calc_energy(x, v)

        x_, v_ = self.leapfrog(x, v)
        x_ = self.wrap(x_)
        h_ = self.calc_energy(x_, v_)

        dh = h_ - h
        exp_mdh = torch.exp(-dh)
        acc = (torch.rand_like(exp_mdh) < exp_mdh).to(DTYPE)
        m = acc[:, None].to(torch.uint8)
        x_ = m * x_.flatten(start_dim=1) + (1 - m) * x.flatten(start_dim=1)
        x_ = x_.reshape(x.shape)
        #  xout, _ = self.flow_forward(x_)
        metrics.update({
            'dt': time.time() - t0,
            'acc': acc,
            'dh': dh,
            'exp_mdh': exp_mdh,
        })
        return x_.detach(), metrics

    def initializer(self, rand: bool = True):
        x = torch.zeros([self.config.nd, ] + self.config.lat)
        if rand:
            x = x.uniform_(0, TWO_PI)

        return x[None, :]

    def lattice_metrics(self, x: torch.Tensor, qold: torch.Tensor):
        q = self._charge_fn(x).detach()
        p = self._plaq_fn(x).detach()
        dq = torch.sqrt((q - qold) ** 2).detach()
        return {'plaq': p, 'q': q, 'dq': dq}

    def run(
            self,
            x: torch.Tensor = None,
            nprint: int = 25,
            nplot: int = 25,
            window: int = 10,
            num_trajs: int = 1024,
            writer: Optional[SummaryWriter] = None,
            plotdir: str = None,
            **kwargs,
    ):
        if x is not None:
            assert isinstance(x, torch.Tensor)

        else:
            x = self.initializer()

        logger.log(f'Running ftHMC with tau={self.tau}, nsteps={self.nstep}')
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
            x_phys, _ = self.flow_forward(x)
            lmetrics = self.lattice_metrics(x_phys, qold)
            metrics = {**metrics_, **lmetrics}

            for key, val in metrics.items():
                if writer is not None:
                    write_summaries(metrics, writer=writer,
                                    step=i, pre='ftHMC')
                try:
                    history[key].append(val)
                except KeyError:
                    history[key] = [val]

            if (i - 1) % nplot == 0 and in_notebook() and plots != {}:
                data = {
                    k: history[k] for k in ['dq', 'acc', 'plaq']
                }
                plotter.update_plots(plots, data, window=window)

            if i % nprint == 0:
                logger.print_metrics(metrics)

        if plotdir is not None and in_notebook():
            plotter.save_live_plots(plots, outdir=plotdir)

        #  histories[n] = history
        # hfile = os.path.join(train_dir, 'train_history.z')
        # io.save_history(history, hfile, name='ftHMC_history')
        plotter.plot_history(history,
                             therm_frac=0.0,
                             outdir=plotdir,
                             config=self.config,
                             lfconfig=self.lfconfig,
                             xlabel='Trajectory')

        return history


def run_ftHMC(
        flow: torch.nn.Module,
        config: TrainConfig,
        tau: float,
        nstep: int
):
    lfconfig = lfConfig(tau=tau, nstep=nstep)
    if torch.cuda.is_available():
        flow.to('cuda')
    flow.eval()
    ft = FieldTransformation(flow=flow,
                             config=config,
                             lfconfig=lfconfig)
    logdir = config.logdir
    ftstr = lfconfig.uniquestr()
    fthmcdir = os.path.join(logdir, 'ftHMC', ftstr)
    pdir = os.path.join(fthmcdir, 'plots')
    sdir = os.path.join(fthmcdir, 'summaries')
    writer = SummaryWriter(log_dir=sdir)
    runs_history = ft.run(nprint=50,
                          figsize=(9., 2.),
                          num_trajs=1024,
                          writer=writer,
                          dpi=120, nplot=10,
                          window=1)
    dirs = {
        'logdir': fthmcdir,
        'plotsdir': pdir,
        'summarydir': sdir,
    }

    return ft, runs_history, dirs
