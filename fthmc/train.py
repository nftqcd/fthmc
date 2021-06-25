"""
train.py

End-to-end training example.
"""
from __future__ import absolute_import, annotations, division, print_function

import os
import time
from dataclasses import asdict, dataclass
from math import pi as PI
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

import fthmc.utils.io as io
import fthmc.utils.logger as logging
import fthmc.utils.plot_helpers as plotter
import fthmc.utils.qed_helpers as qed
from fthmc.config import Param, SchedulerConfig, TrainConfig, FlowModel
from fthmc.utils.distributions import MultivariateUniform, calc_dkl, calc_ess
from fthmc.utils.layers import (get_nets, make_net_from_layers,
                                make_u1_equiv_layers, set_weights)
from fthmc.utils.samplers import BasePrior, apply_flow_to_prior

logger = logging.Logger()
TWO_PI = 2 * PI


@dataclass
class State:
    x: torch.Tensor
    p: torch.Tensor


@dataclass
class qedMetrics:
    param: Param
    plaq: torch.Tensor
    charge: torch.Tensor

    def __post_init__(self):
        self._metrics = {
            'plaq': self.plaq,
            'charge': self.charge
        }


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def list_to_arr(x: list):
    return np.array([grab(torch.stack(i)) for i in x])


def list_to_tensor(x: list[Union[torch.Tensor, np.ndarray]]):
    if isinstance(x[0], torch.Tensor):
        return torch.Tensor(torch.stack(x)).squeeze()
    if isinstance(x[0], np.ndarray):
        return torch.from_numpy(np.stack(x)).squeeze()

    return torch.tensor([grab(torch.stack(i)) for i in x])


def get_model(param: Param, config: TrainConfig):
    prior = MultivariateUniform(torch.zeros((2, *param.lat)),
                                TWO_PI * torch.ones(tuple(param.lat)))
    layers = make_u1_equiv_layers(lattice_shape=tuple(param.lat),
                                  n_layers=config.n_layers,
                                  n_mixture_comps=config.n_s_nets,
                                  hidden_sizes=config.hidden_sizes,
                                  kernel_size=config.kernel_size)
    set_weights(layers)

    #  return {'layers': layers, 'prior': prior}
    return FlowModel(prior=prior, layers=layers)


def restore_model_from_checkpoint(
        infile: Union[str, Path],
        param: Param,
        train_config: TrainConfig,
):
    logger.log(f'Loading checkpoint from: {infile}')
    checkpoint = torch.load(infile)
    model = get_model(param, train_config)
    optimizer = optim.AdamW(
        model.layers.parameters(),
        lr=train_config.base_lr,
        weight_decay=1e-5,
    )

    model.layers.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {'model': model, 'optimizer': optimizer}



def update_history(
        history: dict[str, list],
        metrics: dict[str, list],
        extras: dict = None
):
    def _check_add(d, k, v):
        if isinstance(v, torch.Tensor):
            v = grab(v)
        try:
            d[k].append(v)
        except KeyError:
            d[k] = [v]

        return d

    if extras is not None:
        for key, val in extras.items():
            history = _check_add(history, key, val)
    for key, val in metrics.items():
        history = _check_add(history, key, val)

    return history


def running_averages(
    history: dict[str, Union[list, np.ndarray]],
    n_epochs: int = 10,
    rename: bool = True,
):
    avgs = {}
    for key, val in history.items():
        val = np.array(val)
        if len(val.shape) > 0:
            avgd = val[-n_epochs:].mean()
        else:
            avgd = val.mean()

        if rename:
            key = f'{key}_avg'

        avgs[key] = avgd


    return avgs


#  ActionFn = Callable[[float], torch.Tensor]
ActionFn = Callable[[torch.Tensor], torch.Tensor]
#  Model = dict[str, Union[nn.Module, nn.ModuleList]]


  #  dict[str, nn.Module] = None,

def train_step(
        model: FlowModel,
        param: Param,
        action: ActionFn,
        optimizer: optim.Optimizer,
        batch_size: int,
        scheduler: Any = None,
        pre_model: FlowModel = None,
        dkl_factor: float = 1.,
        scaler: GradScaler = None,
        xi: torch.Tensor = None,
):
    """Perform a single training step.

    TODO: Add `torch.device` to arguments for DDP.
    """
    t0 = time.time()
    #  layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()

    loss_dkl = torch.tensor(0.0)
    if torch.cuda.is_available():
        loss_dkl = loss_dkl.cuda()

    if pre_model is not None:
        #  pre_layers, pre_prior = pre_model['layers'], pre_model['prior']
        #  pre_xi = pre_model['prior'].sample_n(batch_size)
        pre_xi = pre_model.prior.sample_n(batch_size)
        x = qed.ft_flow(pre_model.layers, pre_xi)
        xi = qed.ft_flow_inv(pre_xi, x)

    x, xi, logq = apply_flow_to_prior(model.prior,
                                      model.layers,
                                      #  nn.ModuleList(model['layers']),
                                      xi=xi, batch_size=batch_size)
    logp = (-1.) * action(x)
    dkl = calc_dkl(logp, logq)

    ess = calc_ess(logp, logq)
    qi = qed.batch_charges(xi)
    q = qed.batch_charges(x)
    plaq = logp / (param.beta * param.volume)
    dq = torch.sqrt((q - qi) ** 2)

    loss_dkl = dkl_factor * dkl
    if scaler is not None:
        scaler.scale(loss_dkl).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dkl.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step(loss_dkl)

    metrics = {
        'dt': time.time() - t0,
        'ess': grab(ess),
        'loss_dkl': grab(loss_dkl),
        'q': grab(q),
        'dq': grab(dq.mean(-1)),
        'plaq': grab(plaq),
    }

    return metrics


PlotData = plotter.LivePlotData


def train(
        param: Param,
        config: TrainConfig,
        model: FlowModel = None,
        pre_model: FlowModel = None,
        scheduler_config: SchedulerConfig = None,
        figsize: tuple = None,
        dpi: int = 120,
        dkl_factor: float = 1.,
        history: dict[str, list] = None,
        weight_decay: float = 0.,
        device: str = None,
        xi: torch.Tensor = None,
):
    """Train the flow model."""
    if figsize is None:
        figsize = (4, 3)

    # ---------------------------------------------------------------
    # TODO: Initialize elements of history as `torch.empty(...)` to
    # pre-allocate space for holding history, then accumulate as
    # history[key][era, epoch] = metric
    #
    # Duplicate approach to `FieldTransformation.run` method as well
    # ---------------------------------------------------------------
    if history is None:
        history = {}

    if model is None:
        model = get_model(param, config)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.prior.to(device)
    model.layers.to(device)

    logdir = io.get_logdir(param, config)
    train_dir = os.path.join(logdir, 'training')
    if os.path.isdir(train_dir):
        train_dir = io.tstamp_dir(train_dir)

    dirs = {
        'logdir': logdir,
        'training': train_dir,
        'plots': os.path.join(train_dir, 'plots'),
        'ckpts': os.path.join(train_dir, 'checkpoints'),
    }
    logging.check_else_make_dir(list(dirs.values()))

    u1_action = qed.BatchAction(param.beta)

    optimizer = optim.AdamW(model.layers.parameters(),
                            lr=config.base_lr,
                            weight_decay=weight_decay)
    scheduler = None
    if scheduler_config is not None:
        schcfg = asdict(scheduler_config)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **schcfg)

    logger.log(f'Scheduler created!')

    #  optimizer_force = None
    #  if force_factor > 0:
    #      lr_force = config.base_lr / 100.0
    #      optimizer_force = optim.AdamW(model['layers'].parameters(),
    #                                    lr=lr_force, weight_decay=weight_decay)

    #  optimizer = optimizer_dkl
    #  if config.with_force and force_factor > 0:
    #      optimizer = optimizer_force

    interactive = logging.in_notebook()
    plots = None
    if interactive:
        plots = plotter.init_plots(config, param, dpi=dpi, figsize=figsize)

    step = 0
    dt = 0.0
    skip = ['q']
    ckpt_files = []

    pre = lambda epoch: [f'epoch: {epoch}']

    plot_freq = min((config.plot_freq, config.n_epoch))
    print_freq = min((config.print_freq, config.n_epoch))

    for era in range(config.n_era):
        t0 = time.time()
        estr = f'ERA={era}, last took: {int(dt // 60)} min {dt%60:.4g} sec'
        logger.rule(estr)
        for epoch in range(config.n_epoch):
            step += 1
            metrics = train_step(model=model,
                                 param=param,
                                 xi=xi,
                                 action=u1_action,
                                 optimizer=optimizer,
                                 batch_size=config.batch_size,
                                 scheduler=scheduler,
                                 pre_model=pre_model,
                                 dkl_factor=dkl_factor)
                                 #  with_force=config.with_force,
                                 #  force_factor=force_factor)

            #  if config.with_force:
            #      metrics = train_step(model, param,
            #                           u1_action,
            #                           optimizer,
            #                           config.batch_size,
            #                           config.with_force,
            #                           pre_model=pre_model,
            #                           dkl_factor=dkl_factor,
            #                           force_factor=force_factor)

            for k, v in metrics.items():
                if k in skip:
                    continue
                try:
                    history[k].append(v)
                except KeyError:
                    history[k] = [v]

            win = min(epoch, 20)
            if step % print_freq == 0:
                #  running_avgs = running_averages(history, win, False)
                logger.print_metrics(metrics,
                                     skip=skip,
                                     pre=['(now)', *pre(epoch)])

                logger.print_metrics(history,
                                     skip=skip,
                                     window=win,
                                     pre=['(avg)', *pre(epoch)])

            if step % plot_freq == 0 and plots is not None:
                epdata = PlotData(history['ess'],
                                  plots['dkl']['plot_obj2'])
                lpdata = PlotData(history['loss_dkl'],
                                  plots['dkl']['plot_obj1'])

                plotter.update_joint_plots(lpdata, epdata,
                                           fig=plots['dkl']['fig'],
                                           display_id=plots['dkl']['display_id'])
                dqdata = PlotData(history['dq'],
                                  plots['ess']['plot_obj1'])
                eqdata = PlotData(history['ess'], plots['ess']['plot_obj2'])
                plotter.update_joint_plots(dqdata, eqdata,
                                           fig=plots['ess']['fig'],
                                           display_id=plots['ess']['display_id'])
                #  if config.with_force:
                #      epdata = PlotData(history['ess'],
                #                        plots['force']['plot_obj2'])
                #      lpdata = PlotData(history['loss_force'],
                #                        plots['force']['plot_obj1'])
                #      plotter.update_joint_plots(lpdata, epdata,
                #                                 plots['force']['display_id'])

        dt = time.time() - t0
        ckpt_file = io.save_checkpoint(era=era,
                                       epoch=epoch,
                                       model=model.layers,  # model['layers'],
                                       outdir=dirs['ckpts'],
                                       history=history,
                                       optimizer=optimizer)
        ckpt_files.append(ckpt_file)


    ckpt_file = io.save_checkpoint(era=config.n_era,
                                   epoch=config.n_epoch + 1,
                                   model=model.layers,  # model['layers'],
                                   history=history,
                                   outdir=dirs['ckpts'],
                                   optimizer=optimizer)

    if plots is not None:
        plotter.save_live_plots(plots, dirs['plots'])

    plotter.plot_history(history, param=param, config=config,
                         skip=['epoch', 'step'],
                         num_chains=2, thin=0,
                         therm_frac=0.0, alpha=0.8,
                         xlabel='Epoch', outdir=dirs['plots'])

    hfile = os.path.join(train_dir, 'train_history.z')
    io.save_history(history, hfile, name='train_history')

    outputs = {
        'plots': plots,
        'dirs': dirs,
        'model': model,
        'history': history,
        'optimizer': optimizer,
        'action': u1_action,
    }

    return outputs


def transfer_to_new_lattice(
        L: int,
        layers: nn.ModuleList,
        param_init: Param,
):
    pdict = asdict(param_init)
    pdict['L'] = L
    param = Param(**pdict)
    flow = make_net_from_layers(nets=get_nets(layers),
                                lattice_shape=tuple(param.lat))

    prior = MultivariateUniform(torch.zeros((2, *param.lat)),
                                TWO_PI * torch.ones(tuple(param.lat)))
    model = FlowModel(prior=prior, layers=flow)

    return {'param': param, 'model': model}
