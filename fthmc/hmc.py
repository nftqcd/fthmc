"""
hmc.py
"""
from __future__ import absolute_import, print_function, division, annotations

from fthmc.utils.plot_helpers import plot_history
from fthmc.config import LOGS_DIR
import os
import torch
from fthmc.config import Param
import fthmc.utils.io as io
from fthmc.utils.logger import Logger, check_else_make_dir, savez
import time

import fthmc.utils.qed_helpers as qed
from fthmc.train import get_observables  # , update_history
from math import pi as PI


logger = Logger()
TWO_PI = 2. * PI


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def run_hmc(
        param: Param,
        #  x: torch.Tensor = None,
        #  keep_fields: bool = True,
        plot_metrics: bool = True,
):
    """Run generic HMC.

    Explicitly, we perform `param.nrun` independent experiments, where each
    experiment consists of generating `param.ntraj` trajectories.
    """
    logdir = os.path.join(LOGS_DIR, 'hmc', param.uniquestr())
    if os.path.isdir(logdir):
        logdir = io.tstamp_dir(logdir)

    plots_dir = os.path.join(logdir, 'plots')
    check_else_make_dir(plots_dir)

    action = qed.BatchAction(param.beta)
    logger.log(repr(param))

    dt_run = 0.
    histories = {}
    run_times = []
    fields_arr = []
    for n in range(param.nrun):
        t0 = time.time()

        hstr = f'RUN: {n}, last took: {int(dt_run//60)} m {dt_run%60:.4g} s'
        logger.rule(hstr)

        x = param.initializer()
        p = (-1.) * action(x[None, :]) / (param.beta * param.volume)
        q = qed.batch_charges(x[None, :])

        logger.print_metrics({'plaq': p, 'q': q})
        xarr = []
        history = {
            'dt': [torch.tensor(0.)],
            'traj': [0],
            'acc': [torch.tensor(1.)],
            'dH': [torch.tensor(0.)],
            'q': [q],
            'dqsq': [torch.tensor(0)],
            'plaq': [p],
        }
        for i in range(param.ntraj):
            t1 = time.time()
            dH, exp_mdH, acc, x = qed.hmc(param, x, verbose=False)

            qold = history['q'][-1]
            qnew = qed.batch_charges(x[None, :])
            dqsq = (int(qnew) - int(qold)) ** 2

            plaq = (-1.) * action(x[None, :]) / (param.beta * param.volume)

            xarr.append(x)

            metrics = {
                'traj': n * param.ntraj + i + 1,
                'dt': time.time() - t1,
                'acc': acc,  # 'True' if acc else 'False',
                'dH': dH,
                'q': int(qnew),
                'dqsq': dqsq,
                'plaq': plaq,
            }
            for k, v in metrics.items():
                try:
                    history[k].append(v)
                except KeyError:
                    history[k] = [v]

            #  if (i - 1) % (param.ntraj // param.nprint) == 0:
            if (i - 1) % param.nprint == 0:
                _ = logger.print_metrics(metrics)

        dt = time.time() - t0
        run_times.append(dt)
        histories[n] = history
        fields_arr.append(xarr)

        if plot_metrics:
            outdir = os.path.join(plots_dir, f'run{n}')
            check_else_make_dir(outdir)
            plot_history(history, param, therm_frac=0.0,
                         xlabel='traj', outdir=outdir)

    run_times_strs = [f'{dt:.4f}' for dt in run_times]
    dt_strs = [f'{dt/param.ntraj:.4f}' for dt in run_times]

    logger.log(f'Run times: {run_times_strs}')
    logger.log(f'Per trajectory: {dt_strs}')

    hfile = os.path.join(logdir, f'hmc_histories.z')
    io.save_history(histories, hfile, name='hmc_histories')

    xfile = os.path.join(logdir, 'hmc_fields_arr.z')
    savez(fields_arr, xfile, name='hmc_fields_arr')

    return fields_arr, histories
