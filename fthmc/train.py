"""
train.py

End-to-end training example.
"""
from __future__ import absolute_import, annotations, division, print_function
from pathlib import Path

import time
#  from dataclasses import asdict, dataclass, field
from math import pi as PI
from typing import Union
#  from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

#  from typing import List
import fthmc.utils.io as io
from fthmc.config import LivePlotData
import fthmc.utils.qed_helpers as qed
from fthmc.utils.distributions import (
    MultivariateUniform, bootstrap, calc_dkl, calc_ess
)
from fthmc.utils.layers import make_u1_equiv_layers, set_weights
from fthmc.utils.param import Param
from fthmc.utils.plot_helpers import (
    update_joint_plots, init_plots, update_plot,
)
from fthmc.utils.samplers import apply_flow_to_prior, make_mcmc_ensemble
from fthmc.config import qedMetrics, State, ActionFn, TrainConfig, ftMetrics


#  LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
NOW = io.get_timestamp('%Y-%m-%d-%H%M%S')
METRIC_NAMES = ['dt', 'accept', 'traj', 'dH', 'expdH', 'plaq', 'charge']

logger = io.Logger()
TWO_PI = 2 * PI

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
        #  action = torch.sum(torch.cos(plaq)) / param.volume
        #  action = qed.action(param, x) / (-param.beta * param.volume)
        charge = qed.topo_charge(x[None, :]).to(torch.int)

    return qedMetrics(param=param, action=action, plaq=plaq, charge=charge)


def get_ft_metrics(
        param: Param,
        state0: State,
        state1: State,
        force: torch.Tensor,
        flow: list
):
    force_norm = torch.linalg.norm(force)
    ft_action = qed.ft_action(param, flow, state1.x)
    p0 = state0.p
    p1 = state1.p
    p0_norm = torch.sum(p0 * p0)
    p1_norm = torch.sum(p1 * p1)
    p_norm = torch.sum(p0 * p1) / torch.sqrt(p0_norm * p1_norm)
    #  ft_metrics = ftMetrics(force_norm, ft_action, p_norm)

    #  return ft_metrics
    return {
        'force_norm': force_norm,
        'ft_action': ft_action,
        'p_norm': p_norm,
    }


def run(
        param: Param,
        x: torch.Tensor = None,
        keep_fields: bool = False,
        logger: io.Logger = None
):
    """Run generic HMC."""
    if x is None:
        x = param.initializer()

    fields_arr = []
    metrics = {k: [] for k in METRIC_NAMES}
    logger.log(repr(param))
    observables = get_observables(param, x)
    logger.print_metrics(observables._metrics)
    history = {}
    for n in range(param.nrun):
        t0 = time.time()
        fields = []
        for i in range(param.ntraj):
            t1 = time.time()
            q0 = qed.batch_charges(x[None, :])
            dH, expdH, acc, x = qed.hmc(param, x, verbose=False)
            q1 = qed.batch_charges(x[None, :])
            metrics = {
                'dt': time.time() - t1,
                'traj': n * param.ntraj + i + 1,
                'accept': 'True' if acc else 'False',
                'dH': dH,
                'expdH': expdH,
                'dq': (q1 - q0) ** 2,
                #  'action': observables.action,
                #  'plaq': observables.plaq,
                #  'charge': observables.charge,
            }
            if (i - 1) % (param.ntraj // param.nprint) == 0:
                _ = logger.print_metrics(metrics)

            fields.append(x)

        #  if keep_fields:
        history = update_history(history, metrics,
                                 extras={'run': n})
        fields_arr.append(fields)

        dt = time.time() - t0
        history['dt'].append(dt)
        #  for key, val in traj_metrics.items():

    dt = history['dt']
    logger.log(f'Run times: {[np.round(t, 6) for t in dt]}')
    logger.log(f'Per trajectory: {[np.round(t / param.ntraj, 6) for t in dt]}')

    if keep_fields:
        return fields_arr, metrics

    return x, metrics


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


def update_ft_metrics(metrics, metrics_):
    for key, val in metrics_.items():
        try:
            metrics[key].append(val.detach().cpu().numpy())
        except KeyError:
            metrics[key] = [val.detach().cpu().numpy()]

    return metrics


def ft_leapfrog(param: Param, flow: list, x: torch.Tensor, p: torch.Tensor):
    metrics = {'force_norm': [], 'ft_action': [], 'p_norm': []}

    dt = param.dt
    x = torch.squeeze(x)
    state = State(x, p)

    x = x + 0.5 * dt * p
    force = qed.ft_force(param, flow, x)
    p = p + (-dt) * force

    state1 = State(x, p)

    metrics_ = get_ft_metrics(param, state, state1, force, flow)
    #  logger.print_metrics(metrics_)
    metrics = update_ft_metrics(metrics, metrics_)

    for _ in range(param.nstep - 1):
        state = State(x, p)
        x = x + dt * p

        metrics_ = get_ft_metrics(param, State(x, p), state, force, flow)
        #  logger.print_metrics(metrics_)
        metrics = update_ft_metrics(metrics, metrics_)

        force = qed.ft_force(param, flow, x)
        p = p + (-dt) * force

        #  x1 = x1 + dt * p1
        #  force = qed.ft_force(param, flow, x1)
        #  p1 = p1 + (-dt) * force

    x = x + 0.5 * dt * p
    metrics_ = get_ft_metrics(param, State(x, p), state, force, flow)
    metrics = update_ft_metrics(metrics, metrics_)
    logger.print_metrics(metrics)

    return State(x, p), metrics


def ft_hmc(param: Param, flow: list, x: torch.Tensor):
    x = torch.squeeze(qed.ft_flow_inv(flow, x))
    p = torch.randn_like(x)
    plaq0 = qed.ft_action(param, flow, x) + 0.5 * torch.sum(p * p)
    state1, _ = ft_leapfrog(param, flow, x, p)

    xr = qed.regularize(state1.x)
    p1_norm = torch.sum(state1.p * state1.p)
    plaq1 = qed.ft_action(param, flow, xr) + 0.5 * p1_norm

    prob = torch.rand([], dtype=torch.float64)
    dH = plaq1 - plaq0
    expdH = torch.exp(-dH)
    acc = prob < expdH
    x_ = xr if acc else x
    field_ = qed.ft_flow(flow, x_)
    return (dH, expdH, acc, field_)


def ft_run(
        param: Param,
        flow: list,
        x: torch.Tensor = None,
        keep_fields: bool = False,
        logger: io.Logger = None
):
    """Run generic HMC."""
    if x is None:
        x = param.initializer()

    fields_arr = []
    metrics = {k: [] for k in METRIC_NAMES}
    logger.log(repr(param))
    observables = get_observables(param, x)
    logger.print_metrics(asdict(observables))
    history = traj_metrics = {}
    for n in range(param.nrun):
        t0 = time.time()
        fields = []
        for i in range(param.ntraj):
            #  x = torch.squeeze(x)[None, :]
            dH, expdH, acc, x = ft_hmc(param, flow, x[None, :])
            observables = get_observables(param, x)
            traj_metrics = {
                'traj': n * param.ntraj + i + 1,
                'accept': acc,
                'dH': dH,
                'expdH': expdH,
                'plaq': observables.plaq,
                'charge': observables.charge,
            }
            if (i + 1) % (param.ntraj // param.nprint) == 0:
                _ = logger.print_metrics(traj_metrics)

            if keep_fields:
                fields.append(x)
        if keep_fields:
            fields_arr.append(fields)

        dt = time.time() - t0
        metrics['dt'].append(dt)
        for key, val in traj_metrics.items():
            metrics[key].append(val)

    dt = metrics['dt']
    logger.log(f'Run times: {dt}')
    logger.log(f'Per trajectory: {[t / param.ntraj for t in dt]}')

    if keep_fields:
        return fields_arr, metrics

    return fields, metrics


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
        #  verbose: bool = False,
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
    dq = (q - qi) ** 2

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
        'q': grab(q),
        'dq': grab(dq),
        'ess': grab(ess),
        'plaq': grab(plaq),
        'loss_dkl': grab(loss_dkl),
        'logp': grab(logp),
        'logq': grab(logq),
    }
    if with_force:
        batch_metrics.update({
            'loss_force': force_factor * grab(loss_force),
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
            avgd = np.mean(val[-n_epochs:])
        else:
            avgd = np.mean(val)

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
    optimizer = optim.Adam(
        model['layers'].parameters(),
        lr=train_config.base_lr
    )

    model['layers'].load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {'model': model, 'optimizer': optimizer}


def train(
        param: Param,
        config: TrainConfig,
        model: dict[str, nn.Module] = None,
        pre_model: dict[str, nn.Module] = None,
        figsize: tuple = (8, 3),
        logger: io.Logger = None,
        force_factor: float = 0.01,
        dkl_factor: float = 1.,
        history: dict[[str], list] = None,
):
    """Train the flow model."""
    if logger is None:
        logger = io.Logger()

    if model is None:
        model = get_model(param, config)

    if torch.cuda.is_available():
        model['layers'].cuda()
        model['prior'].cuda()

    u1_action = qed.BatchAction(param.beta)

    optimizer_kdl = optim.Adam(model['layers'].parameters(),
                               lr=config.base_lr, weight_decay=1e-5)
    lr_force = config.base_lr / 100.0
    optimizer_force = optim.Adam(model['layers'].parameters(),
                                 lr=lr_force, weight_decay=1e-5)

    if history is None:
        history = {}

    interactive = io.in_notebook()
    plots = {}
    if interactive:
        plots = init_plots(config, param, figsize=figsize)

    dt = 0.0
    line = (io.WIDTH // 4) * '-'
    for era in range(config.n_era):
        t0 = time.time()
        tprev = f'last took: {int(dt // 60)} min {dt % 60:.4g} s'
        logger.log('\n'.join([line, f'ERA={era}, {tprev}', line]))
        for epoch in range(config.n_epoch):
            metrics = train_step(model, param,
                                 u1_action,
                                 optimizer_kdl,
                                 config.batch_size,
                                 config.with_force,
                                 pre_model=pre_model,
                                 dkl_factor=dkl_factor,
                                 force_factor=force_factor)

            if config.with_force:
                metrics = train_step(model, param,
                                     u1_action,
                                     optimizer_force,
                                     config.batch_size,
                                     config.with_force,
                                     pre_model=pre_model,
                                     dkl_factor=dkl_factor,
                                     force_factor=force_factor)

            step_info = {'epoch': int(epoch), 'step': int(epoch * era)}
            history = update_history(history, metrics, extras=step_info)

            if (epoch + 1) % config.print_freq == 0:
                running_avgs = running_averages(history,
                                                n_epochs=min(epoch, 5))
                logger.print_metrics(running_avgs, skip=['q'])

            if (epoch + 1) % config.plot_freq == 0 and interactive:
                #  dq = np.array(history['dq'])
                #  window = min((epoch, 10))

                #  dq = np.array(history['dq'])[-window:]
                update_plot(y=np.array(history['dq']).mean(axis=-1),
                            ax=plots['dq']['ax'],
                            fig=plots['dq']['fig'],
                            line=plots['dq']['line'],
                            display_id=plots['dq']['display_id'])

                loss_data = LivePlotData(history['loss_dkl'],
                                         plots['dkl']['plot_obj1'])
                ess_data = LivePlotData(history['ess'],
                                        plots['dkl']['plot_obj2'])
                update_joint_plots(loss_data, ess_data,
                                   plots['dkl']['display_id'])

                if config.with_force:
                    loss_force_data = LivePlotData(history['loss_force'],
                                                   plots['force']['plot_obj1'])
                    ess_data = LivePlotData(history['ess'],
                                            plots['force']['plot_obj2'])
                    update_joint_plots(loss_force_data, ess_data,
                                       plots['force']['display_id'])
        dt = time.time() - t0

    outputs = {
        'plots': plots,
        'model': model,
        'history': history,
        'optimizer_kdl': optimizer_kdl,
        'optimizer_force': optimizer_force,
        'action': u1_action,
    }

    return outputs


