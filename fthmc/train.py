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
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

import fthmc.utils.io as io
import fthmc.utils.logger as logging
import fthmc.utils.plot_helpers as plotter
import fthmc.utils.qed_helpers as qed
from fthmc.config import DEVICE, Param, SchedulerConfig, TrainConfig, FlowModel
from fthmc.utils.distributions import MultivariateUniform, calc_dkl, calc_ess
from fthmc.utils.layers import (get_nets, make_net_from_layers,
                                make_u1_equiv_layers, set_weights)
from fthmc.utils.samplers import BasePrior, apply_flow_to_prior

logger = logging.Logger()
TWO_PI = 2 * PI
PlotObject = plotter.PlotObject


# pylint:disable=missing-function-docstring


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


def get_model(config: TrainConfig):
    #  x = TWO_PI * torch.rand(
    #  x = torch.zeros(tuple(config.lat)).uni
    #  prior = MultivariateUniform(torch.zeros((2, *config.lat)),
    #                              TWO_PI * torch.ones(tuple(config.lat)))
    #  x0 = -PI * torch.ones((2, *config.lat))
    #  x1 = PI * torch.ones(tuple(config.lat))
    prior = MultivariateUniform(-PI * torch.ones((2, *config.lat)),
                                PI * torch.ones(tuple(config.lat)))
    layers = make_u1_equiv_layers(lattice_shape=tuple(config.lat),
                                  n_layers=config.n_layers,
                                  n_mixture_comps=config.n_s_nets,
                                  hidden_sizes=config.hidden_sizes,
                                  kernel_size=config.kernel_size,
                                  activation_fn=config.activation_fn)
    set_weights(layers)

    return FlowModel(prior=prior, layers=layers)


def restore_model_from_checkpoint(
        infile: Union[str, Path],
        train_config: TrainConfig,
):
    logger.log(f'Loading checkpoint from: {infile}')
    checkpoint = torch.load(infile)
    model = get_model(train_config)
    optimizer = optim.AdamW(
        model.layers.parameters(),
        lr=train_config.base_lr,
        weight_decay=1e-5,
    )

    model.layers.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {'model': model, 'optimizer': optimizer}


def calc_grads(x: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]):
    x = x.detach().requires_grad_()
    fx = f(x)
    dfdx = torch.autograd.grad(fx, x)[0]
    # For removing GPU memor for large datasets
    del x, fx, f
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dfdx


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


PlotData = plotter.LivePlotData

def update_plots(history: dict, plots: dict):
    epdata = PlotData(history['ess'],
                      plots['dkl']['plot_obj2'])
    lpdata = PlotData(history['loss_dkl'],
                      plots['dkl']['plot_obj1'])
    fig_dkl = plots['dkl']['fig']
    id_dkl = plots['dkl']['display_id']
    plotter.update_joint_plots(lpdata, epdata,
                               fig=fig_dkl, display_id=id_dkl)
    dqdata = PlotData(np.stack(history['dq']).mean(-1),
                      plots['ess']['plot_obj1'])
    eqdata = PlotData(history['ess'], plots['ess']['plot_obj2'])

    fig_ess = plots['ess']['fig']
    id_ess = plots['ess']['display_id']
    plotter.update_joint_plots(dqdata, eqdata,
                               fig=fig_ess, display_id=id_ess)


def write_summaries(metrics: dict, writer: SummaryWriter, step: int):
    for key, val in metrics.items():
        vt = torch.tensor(val)
        if len(vt.shape) > 1:
            writer.add_histogram(f'training/{key}', vt, global_step=step)
        else:
            writer.add_scalar(f'training/{key}', vt.mean(), global_step=step)


ActionFn = Callable[[torch.Tensor], torch.Tensor]


# pylint:disable=invalid-name
def train_step(
        model: FlowModel,
        config: TrainConfig,
        action: ActionFn,
        optimizer: optim.Optimizer,
        batch_size: int,
        scheduler: Any = None,
        scaler: GradScaler = None,
        pre_model: FlowModel = None,
        dkl_factor: float = 1.,
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
        pre_xi = pre_model.prior.sample_n(batch_size)
        x = qed.ft_flow(pre_model.layers, pre_xi)
        xi = qed.ft_flow_inv(pre_model.layers, x)

    #  with torch.cuda.amp.autocast():
    x, xi, logq = apply_flow_to_prior(model.prior,
                                      model.layers,
                                      xi=xi, batch_size=batch_size)
    logp = (-1.) * action(x)
    dkl = calc_dkl(logp, logq)

    ess = calc_ess(logp, logq)
    qi = qed.batch_charges(xi)
    q = qed.batch_charges(x)
    plaq = logp / (config.beta * config.volume)
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
        'logp': grab(logp),
        'logq': grab(logq),
        'loss_dkl': grab(loss_dkl),
        'q': grab(q),
        'dq': grab(dq),
        'plaq': grab(plaq),
    }

    return metrics

OPT_DICT = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
}


def train(
        #  param: Param,
        config: TrainConfig,
        model: FlowModel = None,
        optimizer: torch.Optimizer = None,
        pre_model: FlowModel = None,
        scheduler_config: SchedulerConfig = None,
        figsize: tuple = None,
        dpi: int = 120,
        dkl_factor: float = 1.,
        history: dict[str, list] = None,
        #  weight_decay: float = 0.,
        device: str = None,
        xi: torch.Tensor = None,
        use_scaler: bool = False,
        #  optstr: str = None
):
    """Train the flow model."""
    if figsize is None:
        figsize = (6, 3)

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
        model = get_model(config)

    if device is None:
        device = DEVICE if torch.cuda.is_available() else 'cpu'

    model.prior.to(device)
    model.layers.to(device)

    #  logdir = io.get_logdir(param, config)
    logdir = config.logdir
    train_dir = os.path.join(logdir, 'training')
    if os.path.isdir(train_dir):
        train_dir = io.tstamp_dir(train_dir)

    dirs = {
        'logdir': str(logdir),
        'training': str(train_dir),
        'plots': str(Path(train_dir).joinpath('plots')),
        'ckpts': str(Path(train_dir).joinpath('checkpoints')),
        'summaries': str(Path(train_dir).joinpath('summaries')),
    }
    logging.check_else_make_dir(list(dirs.values()))
    writer = SummaryWriter(log_dir=str(dirs['summaries']))
    logger.log(f'Writing summaries to: {dirs["summaries"]}')
    #  writer.add_graph(model.layers, input_to_model=verbose=True)
    #  writer.add_hparams(asdict(config))

    u1_action = qed.BatchAction(config.beta)
    if optimizer is None:
        optimizer = optim.Adam(model.layers.parameters(), lr=config.base_lr)

    #  if optstr is None:
    #      optimizer = optim.AdamW(model.layers.parameters(),
    #                              lr=config.base_lr,
    #                              weight_decay=weight_decay)
    #  else:
    #      opt = OPT_DICT.get(optstr, None)   # type: optim.Optimizer
    #      if opt is not None and isinstance(opt, optim.Optimizer):
    #          optimizer = opt(model.layers.parameters(), lr=config.base_lr)
    #      else:
    #          logger.log(f'Invalid `optstr={optstr}` specified. '
    #                     'Falling back to default (optim.AdamW)')
    #          optimizer = optim.AdamW(model.layers.parameters(),
    #                                  lr=config.base_lr,
    #                                  weight_decay=weight_decay)

    scheduler = None
    if scheduler_config is not None:
        schcfg = asdict(scheduler_config)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **schcfg)

    logger.log('Scheduler created!')

    scaler = None
    if use_scaler and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.log('Using `GradScaler!')

    #  optimizer_force = None
    #  if force_factor > 0:
    #      lr_force = config.base_lr / 100.0
    #      optimizer_force = optim.AdamW(model['layers'].parameters(),
    #                                    lr=lr_force,
    #                                    weight_decay=weight_decay)
    #  optimizer = optimizer_dkl
    #  if config.with_force and force_factor > 0:
    #      optimizer = optimizer_force

    interactive = logging.in_notebook()
    plots = {'dkl': {}, 'ess': {}}  # type:Union[dict[str, dict[str, Any]]]
    if interactive:
        plots = plotter.init_plots(config, dpi=dpi, figsize=figsize)

    step = 0
    dt = 0.0
    skip = ['q']
    ckpt_files = []

    pre = lambda epoch: [f'epoch={epoch}']  # noqa:E731

    plot_freq = min((config.plot_freq, config.n_epoch))
    print_freq = min((config.print_freq, config.n_epoch))
    log_freq = min((config.log_freq, config.n_epoch))

    for era in range(config.n_era):
        t0 = time.time()
        estr = f'ERA={era}, last took: {int(dt // 60)} min {dt%60:.4g} sec'
        logger.rule(estr)
        for epoch in range(config.n_epoch):
            step += 1
            metrics = train_step(model=model,
                                 config=config,
                                 xi=xi,
                                 action=u1_action,
                                 optimizer=optimizer,
                                 batch_size=config.batch_size,
                                 scaler=scaler,
                                 scheduler=scheduler,
                                 pre_model=pre_model,
                                 dkl_factor=dkl_factor)

            for k, v in metrics.items():
                try:
                    history[k].append(v)
                except KeyError:
                    history[k] = [v]

            if step % log_freq == 0:
                write_summaries(metrics, writer, step)

            if step % print_freq == 0:
                logger.print_metrics(metrics,
                                     skip=skip,
                                     pre=[*pre(step)])

                #  running_avgs = running_averages(history, win, False)
                #  logger.print_metrics(history,
                #                       skip=skip,
                #                       window=10,
                #                       pre=['(avg)', *pre(step)])

            if step % plot_freq == 0 and plots is not None and interactive:
                update_plots(history, plots)

        dt = time.time() - t0
        ckpt_file = io.save_checkpoint(era=era,
                                       epoch=epoch,
                                       model=model.layers,
                                       outdir=dirs['ckpts'],
                                       history=history,
                                       optimizer=optimizer)
        ckpt_files.append(ckpt_file)


    ckpt_file = io.save_checkpoint(era=config.n_era,
                                   epoch=config.n_epoch + 1,
                                   model=model.layers,
                                   history=history,
                                   outdir=dirs['ckpts'],
                                   optimizer=optimizer)

    if plots is not None:
        plotter.save_live_plots(plots, dirs['plots'])

    plotter.plot_history(history, config=config,
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
        'writer': writer,
        'optimizer': optimizer,
        'action': u1_action,
    }

    return outputs


def transfer_to_new_lattice(
        L_new: int,
        layers: nn.ModuleList,
        config: TrainConfig,
        #  param_init: Param,
):
    config_ = asdict(config)
    config_['L'] = L_new
    config_new = TrainConfig(**config_)
    logdir = os.path.join(config.logdir,
                          f'transferred_{L_new}x{L_new}',
                          f'beta{config_new.beta}',
                          config_new.uniquestr())
    dirs = config_new.update_logdirs(logdir)
    flow = make_net_from_layers(nets=get_nets(layers),
                                lattice_shape=tuple(config_new.lat))

    prior_new = MultivariateUniform(-PI * torch.ones((2, *config_new.lat)),
                                    PI * torch.ones(tuple(config_new.lat)))
    model_new = FlowModel(prior=prior_new, layers=flow)

    return {'config': config_new, 'model': model_new}
