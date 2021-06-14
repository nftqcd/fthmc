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
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

import fthmc.utils.io as io
import fthmc.utils.logger as logging
import fthmc.utils.plot_helpers as plotter
import fthmc.utils.qed_helpers as qed
from fthmc.config import Param, TrainConfig
from fthmc.utils.distributions import MultivariateUniform, calc_dkl, calc_ess
from fthmc.utils.layers import (get_nets, make_net_from_layers,
                                make_u1_equiv_layers, set_weights)
from fthmc.utils.samplers import apply_flow_to_prior

#  logger = io.Logger()
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


def list_to_tensor(x: list):
    return torch.tensor([grab(torch.stack(i)) for i in x])


def get_observables(param: Param, x: torch.Tensor):
    x.squeeze()
    if len(x.shape) == 4:
        d = x.shape[1]
        plaq = qed.batch_plaqs(x, 0, 1)
        charge = qed.batch_charges(plaqs=plaq)
        action = torch.sum(torch.cos(plaq), dim=(tuple(range(1, d+1))))
        action /= param.volume
    else:
        plaq = qed.plaq_phase(x)
        plaqsum = torch.sum(torch.cos(qed.plaq_phase(x)))
        action = plaqsum / param.volume

        action = (-param.beta) * plaqsum / (-param.beta * param.volume)
        charge = qed.topo_charge(x[None, :]).to(torch.int)

    return qedMetrics(param=param, plaq=plaq, charge=charge)



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


ActionFn = Callable[[float], torch.Tensor]

def train_step(
        model: dict[str, nn.Module],
        param: Param,
        action: ActionFn,
        optimizer: optim.Optimizer,
        batch_size: int,
        with_force: bool = False,
        pre_model: dict[str, nn.Module] = None,
        force_factor: float = 1.,
        dkl_factor: float = 1.,
        scaler: GradScaler = None,
):
    """Perform a single training step.

    TODO: Add `torch.device` to arguments for DDP.
    """
    t0 = time.time()
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()

    xi = None
    if pre_model is not None:
        pre_layers, pre_prior = pre_model['layers'], pre_model['prior']
        pre_xi = pre_prior.sample_n(batch_size)
        x = qed.ft_flow(pre_layers, pre_xi)
        xi = qed.ft_flow_inv(layers, x)

    xi, x, logq = apply_flow_to_prior(prior, layers, xi=xi,
                                      batch_size=batch_size)
    logp = (-1.) * action(x)
    dkl = calc_dkl(logp, logq)

    #  loss = torch.tensor(0.0)
    loss_dkl = torch.tensor(0.0)
    loss_force = torch.tensor(0.0)
    if torch.cuda.is_available():
        loss_dkl.cuda()
        loss_force.cuda()

    ess = calc_ess(logp, logq)
    qi = qed.batch_charges(xi)
    q = qed.batch_charges(x)
    plaq = logp / (param.beta * param.volume)
    #  action = action(x) / (-param.beta * param.volume)
    dqsq = (q - qi) ** 2

    if with_force:
        assert pre_model is not None
        force = qed.ft_force(param, layers, xi, True)
        force_norm = torch.linalg.norm(force)
        force_size = torch.sum(torch.square(force))
        loss_force = force_factor * force_size
        if scaler is not None:
            scaler.scale(loss_force).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_force.backward()
    else:
        loss_dkl = dkl_factor * dkl
        if scaler is not None:
            scaler.scale(loss_dkl).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dkl.backward()

    optimizer.step()
    loss = loss_dkl + loss_force
    #  loss.backward()
    #  optimizer.step()

    batch_metrics = {
        'dt': time.time() - t0,
        'loss': grab(loss),
        'ess': grab(ess),
        'logp': grab(logp),
        'logq': grab(logq),
        'q': grab(q),
        'dqsq': grab(dqsq),
        'plaq': grab(plaq),
        #  'loss_dkl': grab(loss_dkl),
    }
    if with_force:
        batch_metrics.update({
            #  'loss_force': force_factor * grab(loss_force),
            'force_size': grab(force_size),
            'force_norm': grab(force_norm),
        })

    return batch_metrics


def running_averages(
    history: dict[str, np.ndarray],
    n_epochs: int = 10,
):
    avgs = {}
    for key, val in history.items():
        val = np.array(val)
        if len(val.shape) > 0:
            avgd = val[-n_epochs:].mean()
            #  avgd = np.mean(val[-n_epochs:])
        else:
            avgd = val.mean()
            #  avgd = np.mean(val)

        avgs[key] = avgd

    return avgs



def get_model(param: Param, config: TrainConfig):
    lattice_shape = tuple(param.lat)
    link_shape = (2, *param.lat)

    prior = MultivariateUniform(torch.zeros((2, *param.lat)),
                                TWO_PI * torch.ones(tuple(param.lat)))
    layers = make_u1_equiv_layers(lattice_shape=lattice_shape,
                                  n_layers=config.n_layers,
                                  n_mixture_comps=config.n_s_nets,
                                  hidden_sizes=config.hidden_sizes,
                                  kernel_size=config.kernel_size)
    set_weights(layers)

    return {'layers': layers, 'prior': prior}


def restore_model_from_checkpoint(
        infile: Union[str, Path],
        param: Param,
        train_config: TrainConfig,
):
    logger.log(f'Loading checkpoint from: {infile}')
    checkpoint = torch.load(infile)
    model = get_model(param, train_config)
    optimizer = optim.AdamW(
        model['layers'].parameters(),
        lr=train_config.base_lr,
        weight_decay=1e-5,
    )

    model['layers'].load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {'model': model, 'optimizer': optimizer}


PlotData = plotter.LivePlotData

def train(
        param: Param,
        config: TrainConfig,
        model: dict[str, nn.Module] = None,
        pre_model: dict[str, nn.Module] = None,
        figsize: tuple = (5, 2),
        force_factor: float = 0.01,
        dkl_factor: float = 1.,
        history: dict[str, list] = None,
        weight_decay: float = 1e-5,
):
    """Train the flow model."""
    if model is None:
        model = get_model(param, config)

    if history is None:
        history = {}

    if torch.cuda.is_available():
        model['layers'].cuda()
        model['prior'].cuda()

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

    optimizer_dkl = optim.AdamW(model['layers'].parameters(),
                                lr=config.base_lr, weight_decay=weight_decay)

    optimizer_force = None
    if force_factor > 0:
        lr_force = config.base_lr / 100.0
        optimizer_force = optim.AdamW(model['layers'].parameters(),
                                      lr=lr_force, weight_decay=weight_decay)

    optimizer = optimizer_dkl
    if config.with_force and force_factor > 0:
        optimizer = optimizer_force

    interactive = logging.in_notebook()
    plots = {}
    if interactive:
        plots = plotter.init_plots(config, param, figsize=figsize)

    dt = 0.0
    step = 0
    #  line = (io.WIDTH // 4) * '-'
    print_freq = min((config.print_freq, config.n_epoch))
    plot_freq = min((config.plot_freq, config.n_epoch))
    ckpt_files = []
    for era in range(config.n_era):
        t0 = time.time()
        estr = f'ERA={era}, last took: {int(dt // 60)} min {dt%60:.4g} sec'
        line = len(estr) * '-'
        logger.log('\n'.join([line, estr, line]))
        for epoch in range(config.n_epoch):
            step += 1
            metrics = train_step(model, param,
                                 u1_action,
                                 optimizer,
                                 config.batch_size,
                                 config.with_force,
                                 pre_model=pre_model,
                                 dkl_factor=dkl_factor,
                                 force_factor=force_factor)

            if config.with_force:
                metrics = train_step(model, param,
                                     u1_action,
                                     optimizer,
                                     config.batch_size,
                                     config.with_force,
                                     pre_model=pre_model,
                                     dkl_factor=dkl_factor,
                                     force_factor=force_factor)

            step_info = {'epoch': int(epoch+3)}  # , 'step': int(step+2)}
            history = update_history(history, metrics, extras=step_info)

            if step % print_freq == 0:
                window = min(epoch, 5)
                running_avgs = running_averages(history, n_epochs=window)
                logger.print_metrics(running_avgs, skip=['q'])

            if step % plot_freq == 0 and interactive:
                #  step_info = {'epoch': int(epoch+1)}  # , 'step': int(step+2)}
                #  history = update_history(history, metrics, extras=step_info)
                #  dq = np.array(history['dq'])
                #  window = min((epoch, 10))

                #  dq = np.array(history['dq'])[-window:]
                #  plot_metrics = {
                #      'dqsq': history['dqsq'],
                #      'loss_dkl': history['loss'],
                #      'ess': history['ess'],
                #  }
                #  pavgs = running_averages(plot_metrics,
                #                           n_epochs=min(epoch, 5))
                #  dqsq_avg = np.array(history['dqsq']).mean(axis=-1)
                window = min(epoch, 5)
                dqsq_avg = np.array(history['dqsq']).mean(axis=-1)
                plotter.update_plot(y=dqsq_avg,
                                    ax=plots['dqsq']['ax'],
                                    fig=plots['dqsq']['fig'],
                                    line=plots['dqsq']['line'],
                                    display_id=plots['dqsq']['display_id'])
                #  ess_avg = history['ess'].mean(axis=-1)
                plot_data = {
                    'ess': PlotData(history['ess'],
                                    plots['dkl']['plot_obj2']),
                    'loss_dkl': PlotData(history['loss'],
                                         plots['dkl']['plot_obj1']),
                }
                plotter.update_joint_plots(plot_data['loss_dkl'],
                                           plot_data['ess'],
                                           plots['dkl']['display_id'])

                if config.with_force:
                    plot_data.update({
                        'ess': PlotData(history['ess'],
                                        plots['force']['plot_obj2']),
                        'loss_force': PlotData(history['loss_force'],
                                               plots['force']['plot_obj1']),
                    })
                    plotter.update_joint_plots(plot_data['loss_force'],
                                               plot_data['ess'],
                                               plots['force']['display_id'])

        dt = time.time() - t0
        #  if outdir is not None:
        ckpt_file = io.save_checkpoint(era=era, epoch=epoch+1,
                                       model=model['layers'],
                                       outdir=dirs['ckpts'],
                                       history=history,
                                       optimizer=optimizer)
        ckpt_files.append(ckpt_file)


    ckpt_file = io.save_checkpoint(era=era, epoch=epoch+1,
                                   model=model['layers'],
                                   history=history,
                                   outdir=dirs['ckpts'],
                                   optimizer=optimizer)

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
        #  'optimizer_kdl': ,
        #  'optimizer_force': optimizer_force,
        'action': u1_action,
    }

    return outputs


def transfer_to_new_lattice(
        L: int,
        layers: nn.ModuleList,
        param_init: Param,
        #  config: TrainConfig
):
    pdict = asdict(param_init)
    pdict['L'] = L
    param = Param(**pdict)
    flow = make_net_from_layers(nets=get_nets(layers),
                                lattice_shape=tuple(param.lat))

    prior = MultivariateUniform(torch.zeros((2, *param.lat)),
                                TWO_PI * torch.ones(tuple(param.lat)))
    model = {'layers': flow, 'prior': prior}

    return {'param': param, 'model': model}
