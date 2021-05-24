"""
io.py

Contains helper functions for file IO.
"""
from __future__ import absolute_import, print_function, division, annotations
from dataclasses import dataclass, asdict
from functools import wraps
import os
import shutil
import joblib
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Dict
import datetime

from fthmc.utils.param import Param


WIDTH, HEIGHT = shutil.get_terminal_size(fallback=(156, 50))

def in_notebook():
    """Simple checker function to see if we're currently in a notebook."""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    return True


def check_else_make_dir(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)


def loadz(infile: str):
    return joblib.load(infile)


def savez(obj: dict, fpath: str, name: str = None, logger=None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if logger is None:
        logger = Logger()
    head, _ = os.path.split(fpath)

    check_else_make_dir(head)

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        logger.log(f'Saving {name} to {os.path.abspath(fpath)}.')
    else:
        logger.log(f'Saving {obj.__class__} to {os.path.abspath(fpath)}.')

    joblib.dump(obj, fpath)


# noqa: E999
# pylint:disable=too-few-public-methods,redefined-outer-name
# pylint:disable=missing-function-docstring,missing-class-docstring
class Console:
    """Fallback console object used as in case `rich` isn't installed."""
    @staticmethod
    def log(s, *args, **kwargs):
        now = get_timestamp('%X')
        print(f'[{now}]  {s}', *args, **kwargs)


class Logger:
    """Logger class for pretty printing metrics during training/testing."""
    def __init__(self, width=None):
        if width is None:
            if in_notebook():
                width = 256
            else:
                width = WIDTH

        try:
            # pylint:disable=import-outside-toplevel
            from rich.console import Console as RichConsole
            from rich.theme import Theme
            theme = None
            if in_notebook():
                theme = Theme({
                    'repr.number': 'bold bright_green',
                    'repr.attrib_name': 'bold bright_magenta'
                })
            console = RichConsole(record=False, log_path=False,
                                  force_jupyter=in_notebook(),
                                  log_time_format='[%X] ',
                                  theme=theme, width=width)
        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.width = width
        self.console = console

    def rule(self, s: str, *args, **kwargs):
        """Print horizontal line."""
        width = kwargs.pop('width', self.width)
        w = self.width - (8 + len(s))
        hw = w // 2
        rule = ' '.join((hw * '-', f'{s}', hw * '-'))
        self.console.log(f'{rule}\n', *args, **kwargs)

    def log(self, s: str, *args, **kwargs):
        """Print `s` using `self.console` object."""
        self.console.log(s, *args, **kwargs)

    def load_metrics(self, infile: str = None):
        """Try loading metrics from infile."""
        return joblib.load(infile)

    def print_metrics(
        self,
        metrics: dict,
        window: int = 0,
        pre: list = None,
        outfile: str = None,
    ):
        """Print nicely formatted string of summary of items in `metrics`."""
        outstr = ' '.join([
            strformat(k, v, window) for k, v in metrics.items()
        ])
        if pre is not None:
            outstr = ' '.join([*pre, outstr])

        self.log(outstr)
        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(outstr)

        return outstr

    def save_metrics(
            self,
            metrics: dict,
            outfile: str = None,
            tstamp: str = None,
    ):
        """Save metrics to compressed `.z.` file."""
        if tstamp is None:
            tstamp = get_timestamp('%Y-%m-%d-%H%M%S')

        if outfile is None:
            outdir = os.path.join(os.getcwd(), tstamp)
            fname = 'metrics.z'

        else:
            outdir, fname = os.path.split(outfile)

        check_else_make_dir(outdir)
        outfile = os.path.join(os.getcwd(), tstamp, 'metrics.z')
        self.log(f'Saving metrics to: {outfile}')
        savez(metrics, outfile, name=fname.split('.')[0])


def rename_with_timestamp(
    outfile: str,
    fstr: str = None,
    verbose: bool = True
):
    logger = Logger()
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
        history: dict[str, np.ndarray],
        outfile: str,
        name: str = None
):
    logger = Logger()
    head, tail = os.path.split(outfile)
    check_else_make_dir(head)
    if os.path.isfile(outfile):
        outfile = rename_with_timestamp(outfile)

    savez(history, outfile, name=name)


def save_model(
        model: Union[nn.Module, dict[str, nn.Module]],
        outfile: str,
):
    logger = Logger()
    head, _ = os.path.split(outfile)
    check_else_make_dir(head)
    logger.log(f'Saving model to: {outfile}')
    if os.path.isfile(outfile):
        outfile = rename_with_timestamp(outfile)

    if isinstance(model, dict) and 'layers' in model:
        torch.save(model['layers'].state_dict(), outfile)
    elif isinstance(model, nn.Module):  # pylint:disable=module-not-found
        torch.save(model.state_dict(), outfile)
    else:
        raise ValueError('Unable to save model.')


def load_model(
        param: Param,
        basedir: str = None,
        name: str = None,
        logger: Logger = None,
        **kwargs,
):
    if logger is None:
        logger = Logger()

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
    logger = Logger()
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


def running_averages(
    history: dict,
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


def strformat(k, v, window: int = 0):
    try:
        v = v.cpu()
    except AttributeError:
        pass

    if isinstance(v, int):
        return f'{str(k)}={int(v)}'

    if isinstance(v, bool):
        return f'{str(k)}=True' if v else f'{str(k)}=False'

    if isinstance(v, torch.Tensor):
        v = v.detach().numpy()

    if isinstance(v, (list, np.ndarray)):
        v = np.array(v)
        if window > 0 and len(v.shape) > 0:
            window = min((v.shape[0], window))
            avgd = np.mean(v[-window:])
        else:
            avgd = np.mean(v)

        return f'{str(k)}={avgd:<4.3f}'

    if isinstance(v, float):
        return f'{str(k)}={v:<4.3f}'
    try:
        return f'{str(k)}={v:<3g}'
    except ValueError:
        return f'{str(k)}={v:<3}'



def print_metrics(
        metrics: dict[str, np.ndarray],
        pre: list[str] = None,
        window: int = 0,
        outfile: str = None,
        logger: Logger = None,
):
    if logger is None:
        logger = Logger()

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


def get_timestamp(fstr: str = None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)
