"""
io.py

Contains helper functions for file IO.
"""
from __future__ import absolute_import, print_function, division
from dataclasses import dataclass, asdict
from functools import wraps
import os
import shutil
import joblib
import torch
import numpy as np
from typing import Union
import datetime

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

    def print_metrics(
        self,
        metrics: dict,
        pre: list = None,
        outfile: str = None,
    ):
        outstr = ' '.join([
            strformat(k, v) for k, v in metrics.items()
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
    ):
        """Save metrics to compressed `.z.` file."""
        if outfile is None:
            tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
            outdir = os.path.join(os.getcwd(), tstamp)
            fname = 'metrics.z'

        else:
            outdir, fname = os.path.split(outfile)

        check_else_make_dir(outdir)
        outfile = os.path.join(os.getcwd(), tstamp, 'metrics.z')

        savez(metrics, outfile, name=fname.split('.')[0])


def check_else_make_dir(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)


def savez(obj: dict, fpath: str, name: str = None, logger=None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if logger is None:
        logger = Logger()
    head, tail = os.path.split(fpath)

    check_else_make_dir(head)

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        logger.log(f'Saving {name} to {os.path.abspath(fpath)}.')
    else:
        logger.log(f'Saving {obj.__class__} to {os.path.abspath(fpath)}.')

    joblib.dump(obj, fpath)


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

    if isinstance(v, torch.Tensor):
        v = v.detach().numpy()

    if isinstance(v, bool):
        v = 'True' if v else 'False'

    if isinstance(v, (list, np.ndarray)):
        v = np.array(v)
        #  v = np.array(v)
        #  v = v.detach().numpy()
        if window > 0 and len(v.shape) > 0:
            window = min((v.shape[0], window))
            avgd = np.mean(v[-window:])
        else:
            avgd = np.mean(v)

        #  outstr = f'{str(k)}={avgd:<5.4g}'

        #  if len(v.shape) > 0:
        #      epochs = min((v.shape[0], 5))
        #      avgd = np.mean(v[-epochs:])
        #  else:
        #      avgd = np.mean(v)

        outstr = f'{str(k)}={avgd:<5.4g}'
    else:
        if isinstance(v, float):
            outstr = f'{str(k)}={v:<5.4g}'
        else:
            try:
                outstr = f'{str(k)}={v:<5g}'
            except ValueError:
                outstr = f'{str(k)}={v:<5}'
    return outstr

def print_metrics(
        metrics: dict,
        pre: list = None,
        logger: Logger = None,
        outfile: str = None,
        window: int = 0,
):
    if logger is None:
        logger = Logger()

    outstr = ' '.join([
        strformat(k, v) for k, v in metrics.items()
    ])

    if pre is not None:
        outstr = ' '.join([*pre, outstr])

    logger.log(outstr)
    if outfile is not None:
        with open(outfile, 'a') as f:
            outfile.write(outstr)

    return outstr


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)
