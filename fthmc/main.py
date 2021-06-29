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
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(here)
if modulepath not in sys.path:
    sys.path.append(modulepath)

import fthmc.utils.io as io
import fthmc.utils.qed_helpers as qed
from fthmc.config import CHAINS_TO_PLOT, DPI, FIGSIZE, KWARGS, NUM_SAMPLES, PI, Param, SchedulerConfig, THERM_FRAC, TrainConfig, ftConfig
from fthmc.field_transformation import FieldTransformation
from fthmc.hmc import run_hmc
from fthmc.train import train
from fthmc.utils.distributions import MultivariateUniform
from fthmc.utils.plot_helpers import plot_history
from fthmc.utils.samplers import make_mcmc_ensemble


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


def fthmc(
    flow: torch.nn.ModuleList,
    config: TrainConfig,
    ftconfig: ftConfig,
    xi: torch.Tensor = None,
    nprint: int = 1,
    nplot: int = 1,
    window: int = 1,
    num_trajs: int = 1024,
):
    logger.rule(f'Running `ftHMC` using trained flow for {num_trajs} trajs')
    if torch.cuda.is_available():
        flow.to('cuda')

    flow.eval()

    ft = FieldTransformation(flow=flow, config=config, ftconfig=ftconfig)
    logdir = config.logdir
    ftstr = ftconfig.uniquestr()
    fthmcdir = os.path.join(logdir, 'ftHMC', ftstr)
    pdir = os.path.join(fthmcdir, 'plots')
    sdir = os.path.join(fthmcdir, 'summaries')
    writer = SummaryWriter(log_dir=sdir)
    history = ft.run(x=xi, writer=writer, plotdir=pdir,
                     nprint=nprint, nplot=nplot,
                     window=window, num_trajs=num_trajs)

    return {'field_transformation': ft, 'history': history}


def train_and_evaluate(
        train_config: TrainConfig,
        scheduler_config: SchedulerConfig = None,
        model: Optional[nn.ModuleList] = None,
        pre_model: Optional[nn.ModuleList] = None,
        dpi: Optional[int] = DPI,
        figsize: Optional[tuple] = FIGSIZE,
        num_samples: Optional[int] = NUM_SAMPLES,
        chains_to_plot: Optional[int] = CHAINS_TO_PLOT,
        therm_frac: Optional[float] = THERM_FRAC,
):
    # ----------------------------------------------------------
    # TODO: Deal with loading / restoring model from checkpoint
    # ----------------------------------------------------------
    logger.rule(', '.join([f'Training FlowModel for {train_config.n_era} era',
                           f'each of {train_config.n_epoch} epochs']))
    logger.log(train_config)
    if scheduler_config is not None:
        logger.log(scheduler_config)

    train_outputs = train(config=train_config,
                          model=model,
                          pre_model=pre_model,
                          dpi=dpi,
                          figsize=figsize,
                          scheduler_config=scheduler_config)

    history = None
    if num_samples > 0:
        logger.rule(f'Using trained model to generate {num_samples} samples')
        action_fn = qed.BatchAction(train_config.beta)
        dirs = train_outputs['dirs']
        logdir = dirs['logdir']
        inf_dir = os.path.join(logdir, 'inference')
        inf_pdir = os.path.join(inf_dir, 'plots')
        nb = train_config.batch_size
        model = train_outputs['model']
        history = make_mcmc_ensemble(model=model,
                                     batch_size=nb,
                                     action_fn=action_fn,
                                     num_samples=num_samples)
        plot_history(history=history, config=train_config,
                     num_chains=chains_to_plot, skip=['epoch', 'x'],
                     therm_frac=therm_frac, xlabel='MC Step', outdir=inf_pdir)

    return {'training': train_outputs, 'inference': history}


def main(
        configs: dict,
        xinit: Optional[torch.Tensor] = None,
        num_fthmc_trajs: int = 1024,
        model: Optional[nn.ModuleList] = None,
        pre_model: Optional[nn.ModuleList] = None,
):
    param = configs.get('param', None)
    ftconfig = configs.get('ftconfig', None)
    train_config = configs.get('train_config', None)
    scheduler_config = configs.get('scheduler_config', None)
    #  model = configs.get('model', None)
    #  pre_model = configs.get('pre_model', None)
    kwargs = configs.get('kwargs', None)
    if kwargs is None:
        kwargs = KWARGS

    hmc_outputs = None
    if param is not None:
        param = Param(**param)
        hmc_outputs = hmc(param, x=xinit)

    if train_config is None:
        raise ValueError(f'Expected `train_config` in configs.')

    train_config = TrainConfig(**train_config)

    if scheduler_config is not None:
        scheduler_config = SchedulerConfig(**scheduler_config)

    if ftconfig is not None:
        ftconfig = ftConfig(**ftconfig)
    else:
        logger.log('Setting `ftConfig.tau, `ftConfig.nstep` using `Param`')
        ftconfig = ftConfig(tau=param.tau, nstep=param.nstep)

    outputs = train_and_evaluate(train_config=train_config,
                                 scheduler_config=scheduler_config,
                                 model=model, pre_model=pre_model, **kwargs)
    fthmc_outputs = fthmc(flow=outputs['training']['model'].layers,
                          num_trajs=num_fthmc_trajs,
                          config=train_config, ftconfig=ftconfig)
    return {
        'hmc': hmc_outputs,
        'training': outputs['training'],
        'inference': outputs['inference'],
        'fthmc': fthmc_outputs
    }


if __name__ == '__main__':
    configs = parse_configs()
    logger.print_dict(configs.__dict__, name='parsed configs')
    main(configs.__dict__)
