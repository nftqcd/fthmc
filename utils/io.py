"""
io.py

Contains helper functions for file IO.
"""
from __future__ import absolute_import, print_function, division
from dataclasses import dataclass, asdict
from functools import wraps
import shutil
from typing import Union
import datetime

TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(156, 50))

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
    def __init__(self):
        try:
            # pylint:disable=import-outside-toplevel
            from rich.console import Console as RichConsole
            console = RichConsole(log_path=False, width=TERM_WIDTH)
        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.console = console

    def log(self, s, *args, **kwargs):
        """Print `s` using `self.console` object."""
        self.console.log(s, *args, **kwargs)



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


def print_metrics(
        metrics: dict,
        pre: list = None,
        logger: Logger = None,
        outfile: str = None,
):
    if logger is None:
        logger = Logger()

    outstr = ' '.join([
        f'{str(k):<5}: {v:<7.4g}' if isinstance(v, (float))
        else f'{str(k):<5}: {v:<7g}' for k, v in metrics.items()
    ])

    if pre is not None:
        outstr = ' '.join([*pre, outstr])

    logger.log(outstr)
    if outfile is not None:
        outfile.write(outstr)


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)
