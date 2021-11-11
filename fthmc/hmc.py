"""
hmc.py
"""
from __future__ import absolute_import, print_function, division, annotations

from fthmc.utils.plot_helpers import init_live_plot, plot_history, update_plots
from fthmc.config import CHAINS_TO_PLOT, DTYPE, LOGS_DIR, THERM_FRAC
import os
import torch
from fthmc.config import Param
import fthmc.utils.io as io
from fthmc.utils.logger import Logger, check_else_make_dir, in_notebook, savez
import time

import fthmc.utils.qed_helpers as qed
from math import pi


logger = Logger()
TWO_PI = 2. * pi


# -----------------------------------------------------------------
# TODO: Rewrite HMC as an object similar to `FieldTransformation`
# -----------------------------------------------------------------


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def init_live_plots(
    param: Param,
    #  config: TrainConfig,
    xlabels: list,
    ylabels: list,
    dpi: int = 120,
    use_title: bool = True,
    figsize: tuple[int, int] = None,
    colors: list = None,
):
    plots = {}
    if colors is None:
        colors = [f'C{i}' for i in range(10)]
    else:
        assert len(colors) == len(ylabels)
    for idx, (xlabel, ylabel) in enumerate(zip(xlabels, ylabels)):
        plots[ylabel] = init_live_plot(dpi=dpi, figsize=figsize,
                                       use_title=use_title,
                                       param=param,  # , config=config,
                                       color=colors[idx],
                                       xlabel=xlabel, ylabel=ylabel)

    return plots


def run_hmc(
        param: Param,
        x: torch.Tensor = None,
        #  keep_fields: bool = True,
        plot_metrics: bool = True,
        figsize: tuple[int, int] = None,
        use_title: bool = True,
        save_data: bool = True,
        #  colors: list = None,
        #  nprint: int = 1,
        nplot: int = 10,
):
    """Run generic HMC.

    Explicitly, we perform `param.nrun` independent experiments, where each
    experiment consists of generating `param.ntraj` trajectories.
    """
    logdir = param.logdir
    if os.path.isdir(logdir):
        logdir = io.tstamp_dir(logdir)

    data_dir = os.path.join(logdir, 'data')
    plots_dir = os.path.join(logdir, 'plots')
    check_else_make_dir([plots_dir, data_dir])

    action = qed.BatchAction(param.beta)
    logger.log(repr(param))

    dt_run = 0.
    histories = {}
    run_times = []
    fields_arr = []
    ylabels = ['acc', 'dq', 'plaq']
    xlabels = len(ylabels) * ['trajectory']
    plots = {}
    if in_notebook():
        plots = init_live_plots(param=param,  figsize=figsize,
                                use_title=use_title, # config=config,
                                xlabels=xlabels, ylabels=ylabels)

    for n in range(param.nrun):
        t0 = time.time()

        hstr = f'RUN: {n}, last took: {int(dt_run//60)} m {dt_run%60:.4g} s'
        logger.rule(hstr)

        x = param.initializer()
        p = (-1.) * action(x) / (param.beta * param.volume)
        q = qed.batch_charges(x)

        logger.print_metrics({'plaq': p, 'q': q})
        xarr = []
        history = {}
        for i in range(param.ntraj):
            t1 = time.time()
            dH, exp_mdH, acc, x = qed.hmc(param, x, verbose=False)

            try:
                qold = history['q'][-1]
            except KeyError:
                qold = q
            qnew = qed.batch_charges(x)
            dq = torch.sqrt((qnew - qold) ** 2)

            plaq = (-1.) * action(x) / (param.beta * param.volume)

            xarr.append(x)

            metrics = {
                'traj': n * param.ntraj + i + 1,
                'dt': time.time() - t1,
                'acc': acc.to(DTYPE),  # 'True' if acc else 'False',
                'dH': dH,
                'plaq': plaq,
                'q': int(qnew),
                'dq': dq,
            }
            for k, v in metrics.items():
                try:
                    history[k].append(v)
                except KeyError:
                    history[k] = [v]

            if (i - 1) % param.nprint == 0:
                _ = logger.print_metrics(metrics)

            if in_notebook() and i % nplot == 0:
                data = {
                    'dq': history['dq'],
                    'acc': history['acc'],
                    'plaq': history['plaq'],
                }
                update_plots(plots, data)

        dt = time.time() - t0
        run_times.append(dt)
        histories[n] = history
        fields_arr.append(xarr)

        if plot_metrics:
            outdir = os.path.join(plots_dir, f'run{n}')
            check_else_make_dir(outdir)
            plot_history(history, param, therm_frac=THERM_FRAC,
                         xlabel='traj', outdir=outdir,
                         num_chains=CHAINS_TO_PLOT)

    run_times_strs = [f'{dt:.4f}' for dt in run_times]
    dt_strs = [f'{dt/param.ntraj:.4f}' for dt in run_times]

    logger.log(f'Run times: {run_times_strs}')
    logger.log(f'Per trajectory: {dt_strs}')

    hfile = os.path.join(logdir, 'hmc_histories.z')
    io.save_history(histories, hfile, name='hmc_histories')

    xfile = os.path.join(logdir, 'hmc_fields_arr.z')
    savez(fields_arr, xfile, name='hmc_fields_arr')

    return fields_arr, histories
