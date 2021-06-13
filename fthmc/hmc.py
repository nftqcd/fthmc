"""
hmc.py
"""

from fthmc.utils.plot_helpers import plot_history
from fthmc.config import LOGS_DIR
import os
from pathlib import Path
from typing import Union
import torch
from fthmc.utils.param import Param
import fthmc.utils.io as io
from fthmc.utils.logger import Logger, check_else_make_dir, savez
import time
import numpy as np

import fthmc.utils.qed_helpers as qed
from fthmc.train import get_observables, METRIC_NAMES, update_history

logger = Logger()

def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()

def run_hmc(
        param: Param,
        x: torch.Tensor = None,
        keep_fields: bool = True,
        save_history: bool = True,
        plot_metrics: bool = True,
):
    """Run generic HMC."""
    if x is None:
        x = param.initializer()

    logdir = os.path.join(LOGS_DIR, 'hmc', param.uniquestr())
    if os.path.isdir(logdir):
        logdir = io.tstamp_dir(logdir)

    plots_dir = os.path.join(logdir, 'plots')
    check_else_make_dir(plots_dir)

    fields_arr = []
    metrics = {k: [] for k in METRIC_NAMES}
    logger.log(repr(param))
    observables = get_observables(param, x)
    logger.print_metrics(observables._metrics)
    action = qed.BatchAction(param.beta)
    history = {}
    for n in range(param.nrun):
        t0 = time.time()
        fields = []
        metrics = {}
        for i in range(param.ntraj):
            t1 = time.time()
            xb = x[None, :]

            q0 = qed.batch_charges(xb)
            dH, expdH, acc, x = qed.hmc(param, x, verbose=False)
            q1 = qed.batch_charges(xb)
            dqsq = (q1 - q0) ** 2,

            logp = (-1.) * action(xb)
            metrics_ = {
                'dt': time.time() - t1,
                'traj': n * param.ntraj + i + 1,
                'accept': 'True' if acc else 'False',
                'dH': dH,
                'expdH': expdH,
                'dqsq': dqsq,
                'q': int(q1),
                'logp': logp,
                'plaq': logp / (param.beta * param.volume),
                #  'action': observables.action,
                #  'plaq': observables.plaq,
                #  'charge': observables.charge,
            }
            for k, v in metrics_.items():
                if isinstance(v, torch.Tensor):
                    v = grab(v)
                try:
                    metrics[k].append(v)
                except KeyError:
                    metrics[k] = [v]

            if (i - 1) % (param.ntraj // param.nprint) == 0:
                _ = logger.print_metrics(metrics_)

            fields.append(x)

        if plot_metrics:
            outdir = os.path.join(plots_dir, f'run{n}')
            check_else_make_dir(outdir)
            plot_history(metrics, param, therm_frac=0.0,
                         xlabel='traj', outdir=outdir)

        #  if keep_fields:
        history = update_history(history, metrics,
                                 extras={'run': n})
        #plot_history(history, param, therm_frac=0.0, xlabel='run
        fields_arr.append(fields)

        dt = time.time() - t0
        history['dt'].append(dt)
        #  for key, val in traj_metrics.items():

    dt = history['dt']
    logger.log(f'Run times: {[np.round(t, 6) for t in dt]}')
    logger.log(
        f'Per trajectory: '
        f'{[np.round(np.array(t) / param.ntraj, 6) for t in dt]}'
    )

    #  if logdir is not None:
    if keep_fields:
        hfile = os.path.join(logdir, f'hmc_history.z')
        io.save_history(history, hfile, name='hmc_history')
        if keep_fields:
            xfile = os.path.join(logdir, 'hmc_fields_arr.z')
            savez(fields_arr, xfile, name='hmc_fields_arr')

    if keep_fields:
        return fields_arr, history

    return x, history
