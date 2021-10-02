"""
logger.py
"""
from __future__ import absolute_import, division, print_function, annotations
from dataclasses import asdict, dataclass, is_dataclass
import os
from pathlib import Path
from typing import Any, Union
import torch
import numpy as np

import joblib
import datetime


WITH_CUDA = torch.cuda.is_available()


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def in_notebook():
    """Simple checker function to see if we're currently in a notebook."""
    if os.environ.get('MPLBACKEND', None) is not None:
        return True

    try:
        from IPython import get_ipython
        try:
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except AttributeError:
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


TensorList = "list[torch.Tensor]"
TensorTuple = "tuple[torch.Tensor]"
TensorArrayLike = "Union[TensorList, TensorTuple, torch.Tensor]"
Scalar = "Union[int, bool, float]"


def moving_average1(x: torch.Tensor, window: int = 10):
    if len(x.shape) > 0 and x.shape[0] < window:
        #  return x.mean(keepdim=True)
        return np.mean(x, keepdims=True)
        #  return np.mean(x, keepdims=True)
    #  return torch.convolution(
    return np.convolve(x, np.ones(window), 'valid') / window


ArrayLike = Union[list, np.ndarray, torch.Tensor]

def running_average1(x: ArrayLike, window: int = 10):
    if isinstance(x, torch.Tensor):
        x = grab(x)

    elif isinstance(x, list):
        try:
            x = np.array(x)
        except ValueError:
            x = torch.Tensor(torch.stack(x)).numpy()

    if len(x.shape) > 0:
        avgd = x[-window:].mean()
    else:
        avgd = x.mean()

    return avgd


def moving_average(x: Union[np.ndarray, torch.Tensor], window: int = 0):
    if len(x.shape) > 0:
        x = x[-window:].mean(-1)

    if len(x.shape) > 0 and x.shape[0] < window or window == 0:
        return np.mean(x, keepdims=True)
        #  return x.mean(keepdims=True)

    try:
        return np.convolve(x, np.ones(window), 'valid') / window
    except:
        return np.convolve(np.stack(x), np.ones(window), 'valid') / window


def strformat(
        k: str,
        v: Union[Scalar, TensorArrayLike],
        window: int = 0
):
    if isinstance(v, tuple) and len(v) == 1:
        v = v[0]

    #  if torch.is_tensor(v):
    if isinstance(v, torch.Tensor):
        if v.requires_grad_:
            v = v.detach()

        if WITH_CUDA:
            v = v.cpu()

        v = v.numpy()  # torch.Tensor

    if isinstance(v, int):
        return f'{str(k)}={int(v)}'

    if isinstance(v, bool):
        return f'{str(k)}=True' if v else f'{str(k)}=False'

    if isinstance(v, float):
        return f'{str(k)}={v:<4.3f}'

    if isinstance(v, (list, np.ndarray, torch.Tensor)):
        if isinstance(v, torch.Tensor):
            v = grab(v)

        else:
            v = np.array(v)

        #  avgd = running_average(v, window)
        if window > 0:
            avgd = moving_average(v, window)
        else:
            avgd = v

        #  if window > 0 and len(v.shape) > 0:
        #      if v.shape[0] < window:
        #          avgd = np.mean(v, keepdims=True)
        #          #  window = min((v.shape[0] - 1, 1))
        #      #
        #      #  avgd = moving_average(v, window=window).mean()
        #      #  avgd = v[-window:].mean()
        #      #  return np.convolve(v, np.ones(window), 'valid') / window
        #  #  if window > 0 and len(v.shape) > 0:
        #  #      window = min((v.shape[0], window))
        #  #      avgd = v[-window:].mean()
        #  else:
        #      avgd = v.mean()
        #
        return f'{str(k)}={avgd.mean():<4.3f}'

    try:
        return f'{str(k)}={v:<3g}'
    except ValueError:
        return f'{str(k)}={v:<3}'


# noqa: E999
# pylint:disable=too-few-public-methods,redefined-outer-name
# pylint:disable=missing-function-docstring,missing-class-docstring
class Console:
    """Fallback console object used as in case `rich` isn't installed."""
    def rule(self, s, *args, **kwargs):
        line = len(s) * '-'
        self.log('\n'.join([line, s, line]), *args, **kwargs)

    @staticmethod
    def log(s, *args, **kwargs):
        now = get_timestamp('%X')
        print(f'[{now}]  {s}', *args, **kwargs)



class Logger:
    """Logger class for pretty printing metrics during training/testing."""
    def __init__(self, theme: dict = None):
        try:
            # pylint:disable=import-outside-toplevel
            from rich.console import Console as RichConsole
            from rich.theme import Theme
            if theme is None:
                if in_notebook():
                    theme = {
                        'repr.number': 'bold #87ff00',
                        'repr.attrib_name': 'bold #ff5fff',
                        'repr.str': 'italic #FFFF00',
                    }


            with_jupyter = in_notebook()
            console = RichConsole(record=False, log_path=False,
                                  force_jupyter=with_jupyter,
                                  force_terminal=(not with_jupyter),
                                  log_time_format='[%X] ',
                                  theme=Theme(theme))#, width=width)

        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.console = console

    def rule(self, s: str, *args, **kwargs):
        """Print horizontal line."""
        self.console.rule(s, *args, **kwargs)

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
        outfile: str = None,
        skip: list[str] = None,
        pre: Union[str, list, tuple] = None,
    ):
        """Print nicely formatted string of summary of items in `metrics`."""
        if skip is None:
            skip = []

        fstrs = [
            strformat(k, v, window) for k, v in metrics.items()
            if k not in skip
        ]
        if pre is not None:
            fstrs = [pre, *fstrs] if isinstance(pre, str) else [*pre] + fstrs

        outstr = ' '.join(fstrs)
        self.log(outstr)
        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(outstr)

        return outstr

    def print_dict(self, d: dict, indent: int = 0, name: str = None):
        kvstrs = []
        pre = indent * ' '
        if name is not None:
            nstr = f'{str(name)}'
            line = len(nstr) * '-'
            kvstrs.extend([pre + nstr, pre + line])

        for key, val in d.items():
            if is_dataclass(val):
                val = asdict(val)
            if isinstance(val, dict):
                strs = self.print_dict(val, indent=indent+2, name=key)
            else:
                strs = pre + '='.join([str(key), str(val)])

            kvstrs.append(strs)

        dstr = '\n'.join(kvstrs)
        self.log(dstr)
        return dstr

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
        self.log(f'Saving metrics to: {os.path.relpath(outdir)}')
        savez(metrics, outfile, name=fname.split('.')[0])

        return


def print_dict(d: dict, indent=0, name: str = None):
    kv_strs = []
    pre = indent * ' '

    if name is not None:
        nstr = f'{str(name)}'
        line = len(nstr) * '-'
        kv_strs.extend([pre + nstr, pre + line])

    for key, val in d.items():
        if is_dataclass(val):
            val = asdict(val)
        if isinstance(val, dict):
            strs = print_dict(val, indent=indent+2, name=key)
        else:
            strs = pre + '='.join([str(key), str(val)])

        kv_strs.append(strs)

    return '\n'.join(kv_strs)


def check_else_make_dir(outdir: Union[str, Path, list, tuple]):
    if isinstance(outdir, (str, Path)) and not os.path.isdir(str(outdir)):
        Logger().log(f'Creating directory: {os.path.relpath(outdir)}')
        os.makedirs(str(outdir))

    elif isinstance(outdir, (tuple, list)):
        _ = [check_else_make_dir(str(d)) for d in outdir]


def loadz(infile: str):
    return joblib.load(infile)


def savez(obj: Any, fpath: str, name: str = None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    head, _ = os.path.split(fpath)

    check_else_make_dir(head)

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        Logger().log(f'Saving {name} to {os.path.relpath(fpath)}.')
    else:
        Logger().log(f'Saving {obj.__class__} to {os.path.relpath(fpath)}.')

    joblib.dump(obj, fpath)
