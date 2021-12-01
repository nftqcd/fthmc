"""
io.py

Contains helper functions for file IO.
"""
from __future__ import absolute_import, annotations, division, print_function

import datetime
import os
import shutil
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fthmc.config import LOGS_DIR, Param, TrainConfig
from fthmc.utils.logger import (Logger, check_else_make_dir, get_timestamp,
                                savez, strformat)

logger = Logger()
WIDTH, HEIGHT = shutil.get_terminal_size(fallback=(156, 50))

PathLike = Union[str, Path]

def get_logdir(param: Param, config: TrainConfig):
    """Returns dictionary of unique directories for a given experiment."""
    logdir = os.path.join(LOGS_DIR, param.uniquestr())
    if config is None:
        return logdir

    logdir = os.path.join(logdir, config.uniquestr())
    check_else_make_dir(logdir)
    return logdir


def tstamp_dir(d, fstr=None):
    tstamp = get_timestamp(fstr)
    td = os.path.join(d, tstamp)
    check_else_make_dir(td)
    return td


def rename_with_timestamp(
    outfile: PathLike,
    fstr: str = None,
    verbose: bool = True
):
    if not os.path.isfile(outfile):
        return outfile

    if fstr is None:
        fstr = '%Y-%m-%d-%H%M%S'

    head, tail = os.path.split(outfile)
    fname, ext = tail.split('.')
    tstamp = get_timestamp(fstr)
    new_fname = f'{fname}_{tstamp}.{ext}'
    outfile = os.path.join(head, new_fname)
    if verbose:
        logger.log('\n'.join([
            f'Existing file found!',
            f'Renaming outfile to: {outfile}',
        ]))

    return outfile


def save_history(
        history: dict[str, Any],
        outfile: PathLike,
        name: str = None
):
    head, tail = os.path.split(outfile)
    check_else_make_dir(head)
    if os.path.isfile(outfile):
        outfile = rename_with_timestamp(outfile)

    savez(history, outfile, name=name)



OptimizerDict = "dict[str, optim.Optimizer]"
OptimizerList = "Union[tuple[optim.Optimizer], list[optim.Optimizer]]"
#  OptimizerObject: Union = [
#      optim.Optimizer,
#      dict[str, optim.Optimizer],
#      Union[tuple[optim.Optimizer], list[optim.Optimizer]],
#  ]

def find_and_load_checkpoint(logdir: Union[str, Path]) -> dict[str]:
    if isinstance(logdir, str):
        logdir = Path(logdir)

    conds = lambda f: (
        f.is_file()
        and 'training' in str(f)
        and 'transferred' not in str(f)
    )
    ckpts = sorted(
        [f for f in logdir.rglob('*.tar') if conds(f)],
        key=os.path.getmtime
    )
    if len(ckpts) > 0:
        logger.log(f'Found checkpoint:\n {ckpts[-1]}')
        return torch.load(ckpts[-1])


def save_checkpoint(
        era: int,
        epoch: int,
        model: nn.Module,
        optimizer: Union[optim.Optimizer, OptimizerDict, OptimizerList],
        outdir: PathLike,  # Union[str, Path],
        history: dict[str, list] = None,
        overwrite: bool = False,
):
    if not isinstance(optimizer, (optim.Optimizer, dict, tuple, list)):
        raise ValueError('Expected `optimizer` to be one of: '
                         '`[optim.Optimizer, dict, tuple, list]`.\n'
                         f'Receieved: {type(optimizer)}')

    fname = '-'.join([
        'ckpt',
        f'era{era}'.zfill(3),
        f'epoch{epoch}'.zfill(5)
    ])
    # fname = f'{fname}.tar'
    # fname = f'ckpt-era{era_str}-epoch{epoch}.tar'
    ckpt_dir = Path(str(outdir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # -------------------------------------------------
    # TODO: Deal with only keeping last N checkpoints
    # -------------------------------------------------
    #  ckpt_files = [os.path.join(ckpt_dir, i) for i in os.listdir(ckpt_dir)]
    outfile = os.path.join(ckpt_dir, fname)
    outfile = str(Path(ckpt_dir).joinpath(f'{fname}.tar'))
    #  outfile = os.path.abspath(str(outfile))
    #  head, _ = os.path.split(outfile)
    #  check_else_make_dir(head)
    #  if os.path.isfile(outfile) and not overwrite:
    #      outfile = rename_with_timestamp(outfile)
    checkpoint = {
        'era': era,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
    }

    if history is not None:
        checkpoint['history'] = history

    if isinstance(optimizer, optim.Optimizer):
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if isinstance(optimizer, (tuple, list)):
        for idx, opt in enumerate(optimizer):
            checkpoint[f'optimizer{idx}_state_dict'] = opt.state_dict()

    if isinstance(optimizer, dict):
        for key, val in optimizer.items():
            checkpoint[f'optimizer_{key}_state_dict'] = val.state_dict()

    logger.log(f'Saving checkpoint to: {outfile}')
    torch.save(checkpoint, outfile)

    return outfile


def load_checkpoint(
        infile: str,
):
    logger.log(f'Loading checkpoint from: {infile}')
    checkpoint = torch.load(infile)

    return checkpoint


def save_model(
        model: Union[nn.Module, dict[str, nn.Module]],
        outfile: str,
):
    head, _ = os.path.split(outfile)
    check_else_make_dir(head)
    if os.path.isfile(outfile):
        outfile = rename_with_timestamp(outfile)

    logger.log(f'Saving model to: {outfile}')
    if isinstance(model, dict) and 'layers' in model:
        torch.save(model['layers'].state_dict(), outfile)
    elif isinstance(model, nn.Module):
        torch.save(model.state_dict(), outfile)
    else:
        raise ValueError('Unable to save model.')


def load_model(
        param: Param,
        basedir: str = None,
        name: str = None,
        **kwargs,
):
    if basedir is None:
        basedir = os.getcwd()

    if name is None:
        name = 'model_state_dict'

    outdir = os.path.join(basedir, param.uniquestr())
    outfile = os.path.join(outdir, f'{name}.pt')
    logger.log(f'Loading model from: {outfile}')
    model = torch.load(outfile, **kwargs)

    return model


def logit(logfile='out.log'):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = ' '.join((func.__name__, 'was called'))
            logger.log(log_string)
            # Open the logfile and append
            with open(logfile, 'a') as f:
                # write to the logfile
                f.write(f'{log_string}\n')
            return func(*args, **kwargs)
        return wrapped_function
    return logging_decorator


def print_metrics(
        metrics: dict[str, np.ndarray],
        pre: list[str] = None,
        window: int = 0,
        outfile: str = None,
):
    outstr = ' '.join([
        strformat(k, v, window=window) for k, v in metrics.items()
    ])

    if pre is not None:
        outstr = ' '.join([*pre, outstr])

    logger.log(outstr)
    if outfile is not None:
        with open(outfile, 'a') as f:
            f.write(outstr)

    return outstr


