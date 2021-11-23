"""
main.py

Executable for training the flow model from the CLI.
"""
from __future__ import absolute_import, annotations, division, print_function

import argparse
import datetime
import json
import os
import sys
from dataclasses import asdict
from typing import Optional
import joblib

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(here)
if modulepath not in sys.path:
    sys.path.append(modulepath)

from fthmc.config import FlowModel
import fthmc.utils.io as io
import fthmc.utils.qed_helpers as qed
from fthmc.config import (CHAINS_TO_PLOT, Param, SchedulerConfig, TrainConfig,
                          lfConfig)
from fthmc.ft_hmc import FieldTransformation
from fthmc.hmc import run_hmc
from fthmc.train import train, transfer_to_new_lattice
# from fthmc.utils.distributions import MultivariateUniform
#  from fthmc.utils.samplers import make_mcmc_ensemble
from fthmc.utils.inference import make_mcmc_ensemble
from fthmc.utils.plot_helpers import plot_history
from fthmc.utils.layers import (get_nets, make_net_from_layers,
                                make_u1_equiv_layers, set_weights)
from fthmc.utils.distributions import MultivariateUniform
from math import pi
#  from fthmc.utils.parse_configs import parse_configs

logger = io.Logger()


def parse_configs():
    """Parse configs from JSON file."""
    parser = argparse.ArgumentParser(
        f'Normalizing flow model on 2D U(1) lattice gauge model.'
    )
    parser.add_argument('--json_file',
                        dest='json_file',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to JSON file containing configuration.')
    args = parser.parse_args()
    with open(args.json_file, 'rt') as f:
        targs = argparse.Namespace()
        targs.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=targs)

    return args


def hmc(param: Param, x: torch.Tensor = None):
    logger.rule(f'Running HMC for {param.nrun} runs, '
                f'each of {param.ntraj} trajectories')
    return run_hmc(param, x=x)


def run_fthmc(
        flow: nn.ModuleList,
        config: TrainConfig,
        lfconfig: lfConfig,
        xi: torch.Tensor = None,
        nprint: int = 50,
        nplot: int = 10,
        window: int = 1,
        num_trajs: int = 1024,
        **kwargs,
):
    logger.rule(f'Running `ftHMC` using trained flow for {num_trajs} trajs')
    if torch.cuda.is_available():
        flow.to('cuda')

    flow.eval()

    ft = FieldTransformation(flow=flow, config=config, lfconfig=lfconfig)
    logdir = config.logdir
    ftstr = lfconfig.uniquestr()
    fthmcdir = os.path.join(logdir, 'ftHMC', ftstr)
    pdir = os.path.join(fthmcdir, 'plots')
    sdir = os.path.join(fthmcdir, 'summaries')
    writer = SummaryWriter(log_dir=sdir)
    history = ft.run(x=xi, nprint=nprint, nplot=nplot, window=window,
                     num_trajs=num_trajs, writer=writer, plotdir=pdir,
                     **kwargs)
    histfile = os.path.join(fthmcdir, 'history.z')
    logger.log(f'Saving history to: {histfile}')
    joblib.dump(history, histfile)

    return {'field_transformation': ft, 'history': history}


def train_and_evaluate(
        train_config: TrainConfig,
        model: FlowModel,
        pre_model: FlowModel = None,
        num_samples: int = 1024,
        lfconfig: lfConfig = None,
        num_fthmc_trajs: int = 1024,
        chains_to_plot: int = CHAINS_TO_PLOT,
        scheduler_config: SchedulerConfig = None,
        **kwargs,
):
    # ----------------------------------------------------------
    # TODO: Deal with loading / restoring model from checkpoint
    # ----------------------------------------------------------
    logger.rule(', '.join([f'Training FlowModel for {train_config.n_era} era',
                           f'each of {train_config.n_epoch} epochs']))
    logger.log(f'{train_config}')
    if scheduler_config is not None:
        logger.log(f'{scheduler_config}')

    train_out = train(config=train_config, model=model,
                      pre_model=pre_model,
                      scheduler_config=scheduler_config)

    history = {}
    if num_samples > 0:
        logger.rule(f'Using trained model to generate {num_samples} samples')
        action_fn = qed.BatchAction(train_config.beta)
        dirs = train_out['dirs']
        logdir = dirs['logdir']
        inf_dir = os.path.join(logdir, 'inference')
        inf_pdir = os.path.join(inf_dir, 'plots')
        history = make_mcmc_ensemble(model=train_out['model'],
                                     action_fn=action_fn,
                                     num_samples=num_samples,
                                     batch_size=train_config.batch_size)
        _ = history.__dict__.pop('skip', None)
        plot_history(history=history, config=train_config,
                     num_chains=chains_to_plot, skip=['epoch', 'x'],
                     xlabel='MC Step', outdir=inf_pdir, **kwargs)

    fthmc_out = {}
    if lfconfig is not None and num_fthmc_trajs > 0:
        flow = train_out['model'].layers
        fthmc_out = run_fthmc(flow=flow,
                              num_trajs=num_fthmc_trajs,
                              config=train_config,
                              lfconfig=lfconfig)

    return {'training': train_out, 'inference': history, 'fthmc': fthmc_out}


def transfer(
        L_new: int,
        config: TrainConfig,
        layers: nn.ModuleList,
        lfconfig: lfConfig,
        param: Param = None,
        new_lr: float = None,
        run_hmc: bool = True,
        figsize: tuple = None,
        num_fthmc_trajs: int = 1024,
        scheduler_config: SchedulerConfig = None,
):
    logger.rule(f'Transferring trained model to {L_new}x{L_new} lattice')
    if figsize is None:
        figsize = (9, 2)

    config_ = asdict(config)
    config_['L'] = L_new
    config_new = TrainConfig(**config_)

    logdir = os.path.join(config.logdir,
                          f'transferred_{L_new}x{L_new}',
                          f'beta{config_new.beta}',
                          config_new.uniquestr())
    _ = config_new.update_logdirs(logdir)

    flow = make_net_from_layers(nets=get_nets(layers),
                                lattice_shape=tuple(config_new.lat))

    prior_new = MultivariateUniform(-pi * torch.ones((2, *config_new.lat)),
                                    pi * torch.ones(tuple(config_new.lat)))
    model_new = FlowModel(prior=prior_new, layers=flow)

    # xfr = transfer_to_new_lattice(L_new=L, layers=layers, config=config)
    # config_new = xfr['config']
    # model_new = xfr['model']
    # logdir = os.path.join(config.logdir,
    #                       f'transferred_{L}x{L}',
    #                       f'beta{config_new.beta}',
    #                       config_new.uniquestr())
    # io.check_else_make_dir(logdir)
    if run_hmc and param is not None:
        param_ = asdict(param)
        param_['L'] = config_new.L
        param_new = Param(**param_)
        hmc_fields, hmc_history = hmc(param_new)

    if new_lr is not None:
        cfg = asdict(config_new)
        cfg['base_lr'] = new_lr
        config_new = TrainConfig(**cfg)

    outputs = train_and_evaluate(config_new,
                                 model=model_new,
                                 figsize=figsize,
                                 lfconfig=lfconfig,
                                 num_fthmc_trajs=num_fthmc_trajs,
                                 scheduler_config=scheduler_config)
    #  flow = train_outputs['tra
    #  fthmc_outputs = run_fthmc(flow=train_outputs['model'].layers,
    #                            num_trajs=num_fthmc_trajs,
    #                            config=config_new, ftconfig=ftconfig)
    return {
        'config': config_new,
        'model': model_new,
        'train_outputs': outputs['training'],
        'inference_outputs': outputs['inference'],
        'fthmc_outputs': outputs['fthmc'],
    }


def setup(
        configs: dict,
):
    train_config = configs.get('train_config', None)
    assert train_config is not None
    train_config = TrainConfig(**train_config)

    param = configs.get('param', None)
    if param is not None:
        param = Param(**param)

    lfconfig = configs.get('lfconfig', None)
    if lfconfig is not None:
        lfconfig = lfConfig(**lfconfig)
    else:
        logger.log('Setting `lfConfig.tau, `lfConfig.nstep` using `Param`')
        assert param is not None
        lfconfig = lfConfig(tau=param.tau, nstep=param.nstep)

    scheduler_config = configs.get('scheduler_config', None)
    #  model = configs.get('model', None)
    #  pre_model = configs.get('pre_model', None)
    #  kwargs = configs.get('kwargs', None)
    #  if kwargs is None:
    #      kwargs = KWARGS

    if scheduler_config is not None:
        scheduler_config = SchedulerConfig(**scheduler_config)

    return {
        'param': param,
        'train_config': train_config,
        'lfconfig': lfconfig,
        'scheduler_config': scheduler_config,
        'kwargs': kwargs
    }


def main(
        config: TrainConfig,
        param: Param = None,
        lfconfig: lfConfig = None,
        pre_model: FlowModel = None,
        scheduler_config: SchedulerConfig = None,
        **kwargs,
):
    num_fthmc_trajs = kwargs.get('num_fthmc_trajs', 1024)
    hmc_out = None if param is None else hmc(param)
    if lfconfig is None:
        assert param is not None
        lfconfig = lfConfig(tau=param.tau, nstep=param.nstep)

    outputs = train_and_evaluate(train_config=config,
                                 lfconfig=lfconfig,
                                 num_fthmc_trajs=1024,
                                 scheduler_config=scheduler_config,
                                 pre_model=pre_model, **kwargs)
    #  fthmc_out = run_fthmc(flow=outputs['training']['model'].layers,
    #                            config=config, ftconfig=ftconfig, **kwargs)

    L_new = 2 * config.L
    layers = outputs['training']['model'].layers
    transfer_out = transfer(L_new=L_new,
                            param=param,
                            layers=layers,
                            lfconfig=lfconfig,
                            config=config,
                            num_fthmc_trajs=num_fthmc_trajs)
    return {
        'hmc': hmc_out,
        'training': outputs['training'],
        'inference': outputs['inference'],
        'fthmc': outputs['fthmc'],
        'transfer': transfer_out,
    }


if __name__ == '__main__':
    configs = parse_configs()
    logger.print_dict(configs.__dict__, name='parsed configs')
    cfgs = setup(configs.__dict__)
    _ = main(config=cfgs['train_config'],
             param=cfgs['param'],
             lfconfig=cfgs['lfconfig'])
    #  main(config=cfgs['train_config'],

    #  main(configs.__dict__)
