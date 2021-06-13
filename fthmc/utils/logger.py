"""
logger.py
"""
from __future__ import absolute_import, division, print_function, annotations
import os
from pathlib import Path
from typing import Union
import torch
import numpy as np

import joblib
import datetime


def in_notebook():
    """Simple checker function to see if we're currently in a notebook."""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    return True



def get_timestamp(fstr: str = None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def strformat(k, v, window: int = 0):
    if isinstance(v, tuple) and len(v) == 1:
        v = v[0]

    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    #  if isinstance(v, torch.Tensor):
    #      v = v.detach().numpy()

    #  try:
    #      v = v.cpu()
    #  except AttributeError:
    #      pass

    if isinstance(v, int):
        return f'{str(k)}={int(v)}'

    if isinstance(v, bool):
        return f'{str(k)}=True' if v else f'{str(k)}=False'

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
                                  theme=theme)#, width=width)
        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.width = width
        self.console = console

    def rule(self, s: str, *args, **kwargs):
        """Print horizontal line."""
        #  width = kwargs.pop('width', self.width)
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
        skip: list[str] = None,
    ):
        """Print nicely formatted string of summary of items in `metrics`."""
        if skip is None:
            skip = []

        fstrs = [
            strformat(k, v, window) for k, v in metrics.items()
            if k not in skip
        ]
        if pre is not None:
            fstrs = [*pre, fstrs]

        outstr = ' '.join(fstrs)
        #  outstr = ' '.join([
        #      strformat(k, v, window) for k, v in metrics.items()
        #      if k not in skip
        #  ])
        #  if pre is not None:
        #      outstr = ' '.join([*pre, outstr])
        #
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


def check_else_make_dir(outdir: Union[str, Path, list, tuple]):
    if isinstance(outdir, (str, Path)) and not os.path.isdir(str(outdir)):
        Logger().log(f'Creating directory: {outdir}')
        os.makedirs(str(outdir))

    elif isinstance(outdir, (tuple, list)):
        _ = [check_else_make_dir(str(d)) for d in outdir]

    #  if not os.path.isdir(outdir):
    #      Logger().log(f'Creating directory: {outdir}')
    #      os.makedirs(outdir)


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





