from __future__ import absolute_import, division, print_function, annotations
from fthmc.utils.logger import in_notebook
import torch
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from fthmc.utils.param import Param
import fthmc.utils.io as io

from fthmc.config import PlotObject, LivePlotData, TrainConfig

from IPython.display import display, DisplayHandle

logger = io.Logger()

PathLike = Union[str, Path]
Metric = Union[list, np.ndarray, torch.Tensor]


def therm_arr(
        x: np.ndarray,
        therm_frac: float = 0.1,
        ret_steps: bool = True
):
    #  taxis = np.argmax(x.shape)
    taxis = 0
    num_steps = x.shape[taxis]
    therm_steps = int(therm_frac * num_steps)
    x = np.delete(x, np.s_[:therm_steps], axis=taxis)
    t = np.arange(therm_steps, num_steps)
    if ret_steps:
        return x, t
    return x


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def list_to_arr(x: list):
    return np.array([grab(torch.stack(i)) for i in x])


def savefig(fig: plt.Figure, outfile: str, dpi: int = 500):
    io.check_else_make_dir(os.path.dirname(outfile))
    outfile = io.rename_with_timestamp(outfile, fstr='%H%M%S')
    logger.log(f'Saving figure to: {outfile}')
    #  fig.clf()
    #  plt.close('all')
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')


def save_live_plots(plots: dict[str, dict], outdir: PathLike):
    if isinstance(outdir, Path):
        outdir = str(outdir)

    io.check_else_make_dir(outdir)
    for key, plot in plots.items():
        outfile = os.path.join(outdir, f'{key}_live.pdf')
        try:
            savefig(plot['fig'], outfile)
        except KeyError:
            continue

    #  dqf = os.path.join(outdir, 'dq_training.pdf')
    #  savefig(plots['dq']['fig'], dqf)
    #
    #  dklf = os.path.join(outdir, 'loss_dkl_ess_training.pdf')
    #  savefig(plots['dkl']['fig'], dklf)
    #
    #  if 'force' in plots and 'fig' in plots['force']:
    #      ff = os.path.join(outdir, 'loss_force_training.pdf')
    #      savefig(plots['force']['fig'], ff)


def plot_metric(
        metric: Metric,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        hline: bool = False,
        thin: int = 0,
        num_chains: int = 10,
        therm_frac: float = 0.,
        outfile: PathLike = None,
        figsize: tuple[int] = None,
        **kwargs,
):
    """Plot metric object."""
    if figsize is None:
        figsize = (4, 3)

    if isinstance(metric, torch.Tensor):
        metric = metric.cpu().numpy()

    elif isinstance(metric, list):
        if len(metric) == 1:
            metric = np.array([grab(metric[0])])
        else:
            if isinstance(metric[0], torch.Tensor):
                metric = list_to_arr(metric)

    metric = np.array(metric)
    if thin > 0:
        metric = metric[::thin]

    metric, steps = therm_arr(metric, therm_frac, ret_steps=True)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if len(metric.squeeze().shape) == 1:
        label = kwargs.pop('label', None)
        ax.plot(steps, metric, label=label, **kwargs)

    if len(metric.squeeze().shape) == 2:
        for idx in range(num_chains):
            y = metric[:, idx]
            ax.plot(steps, y, **kwargs)

    if hline:
        avg = np.mean(metric)
        label = f'avg: {avg:.4g}'
        ax.axhline(avg, label=label, **kwargs)

    ax.grid(True, alpha=0.5)
    if xlabel is None:
        xlabel = 'Step'

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if outfile is not None:
        savefig(fig, outfile)

    return fig, ax



def plot_history(
        history: dict[str, np.ndarray],
        param: Param,
        therm_frac: float = 0.0,
        config: TrainConfig = None,
        xlabel: str = None,
        title: str = None,
        num_chains: int = 10,
        outdir: str = None,
        skip: list[str] = None,
        thin: int = 0,
        hline: bool = False,
        **kwargs,
):
    for key, val in history.items():
        if skip is not None and key in skip:
            continue

        outfile = None
        if outdir is not None:
            outfile = os.path.join(outdir, f'{key}.pdf')

        if title is None:
            tarr = get_title(param, config)
            title = '\n'.join(tarr) if len(tarr) > 0 else None

        _ = plot_metric(val,
                        ylabel=key,
                        title=title,
                        xlabel=xlabel,
                        outfile=outfile,
                        thin=thin,
                        therm_frac=therm_frac,
                        num_chains=num_chains, **kwargs)


def init_plots(config: TrainConfig, param: Param, figsize: tuple = (8, 3)):
    plots_dqsq = {}
    plots_dkl = {}
    plots_force = {}
    if in_notebook:
        plots_dqsq = init_live_plot(figsize=figsize,
                                    param=param, config=config,
                                    ylabel='dqsq', xlabel='Epoch')

        ylabel_dkl = ['loss_dkl', 'ESS']
        plots_dkl = init_live_joint_plots(config.n_era, config.n_epoch,
                                          figsize=figsize, param=param,
                                          config=config,
                                          ylabel=ylabel_dkl)

        if config.with_force:
            ylabel_force = ['loss_force', 'ESS']
            plots_force = init_live_joint_plots(config.n_era,
                                                config.n_epoch,
                                                figsize=figsize,
                                                param=param,
                                                config=config,
                                                ylabel=ylabel_force)

    return {
        'dqsq': plots_dqsq,
        'dkl': plots_dkl,
        'force': plots_force,
    }


def init_live_joint_plots(
        n_era: int,
        n_epoch: int,
        dpi: int = 400,
        figsize: tuple = (5, 2),
        param: Param = None,
        config: TrainConfig = None,
        xlabel: str = None,
        ylabel: list[str] = None,
        colors: list[str] = None,
):
    if colors is None:
        colors = ['#0096ff', '#f92672']

    #  sns.set_style('ticks')
    fig, ax0 = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                            constrained_layout=True)
    plt.xlim(0, n_era * n_epoch)
    line0 = ax0.plot([0], [0], alpha=0.5, c=colors[0])
    ax1 = ax0.twinx()
    if ylabel is None:
        ax0.set_ylabel('Loss', color=colors[0])
        ax1.set_ylabel('ess', color=colors[1])

    else:
        ax0.set_ylabel(ylabel[0], color=colors[0])
        ax1.set_ylabel(ylabel[1], color=colors[1])

    ax0.tick_params(axis='y', labelcolor=colors[0])
    ax0.grid(False)

    line1 = ax1.plot([0], [0], alpha=0.8, c=colors[1])  # dummy

    ax0.tick_params(axis='y', labelcolor=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[1])
    ax1.grid(False)
    ax0.set_xlabel('Epoch' if xlabel is None else xlabel)

    title = get_title(param, config)
    if len(title) > 0:
        fig.suptitle('\n'.join(title))
    #  title = ''
    #  if param is not None:
    #      title += param.uniquestr()
    #  if config is not None:
    #      title += config.uniquestr()

    #  fig.suptitle(title)

    display_id = display(fig, display_id=True)
    plot_obj1 = PlotObject(ax0, line0)
    plot_obj2 = PlotObject(ax1, line1)
    return {
        'fig': fig,
        'ax0': ax0,
        'ax1': ax1,
        'plot_obj1': plot_obj1,
        'plot_obj2': plot_obj2,
        'display_id': display_id
    }


def get_title(param: Param = None, config: TrainConfig = None):
    title = []
    if param is not None:
        title.append(param.uniquestr())
    if config is not None:
        title.append(config.uniquestr())

    return title


def init_live_plot(
        dpi: int = 400,
        figsize: tuple[int] = (5, 2),
        param: Param = None,
        config: TrainConfig = None,
        xlabel: str = None,
        ylabel: str = None,
        **kwargs
):
    color = kwargs.pop('color', '#0096FF')
    xlabel = 'Epoch' if xlabel is None else xlabel
    #  sns.set_style('ticks')
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize, constrained_layout=True)
    line = ax.plot([0], [0], c=color, **kwargs)

    title = get_title(param, config)

    if len(title) > 0:
        _ = fig.suptitle('\n'.join(title))

    if ylabel is not None:
        _ = ax.set_ylabel(ylabel, color=color)

    ax.tick_params(axis='y', labelcolor=color)

    _ = ax.autoscale(True, axis='y')
    #  plt.Axes.autoscale(True, axis='y')
    display_id = display(fig, display_id=True)
    return {
        'fig': fig, 'ax': ax, 'line': line, 'display_id': display_id,
    }


def moving_average(x: np.ndarray, window: int = 10):
    #  if len(x) < window:
    if len(x.shape) > 0 and x.shape[0] < window:
        return np.mean(x, keepdims=True)
    #  if x.shape[0] < window:
    #      return np.mean(x, keepdims=True)

    return np.convolve(x, np.ones(window), 'valid') / window


def update_plot(
        y: Metric,
        fig: plt.Figure,
        ax: plt.Axes,
        line: list[plt.Line2D],
        display_id: DisplayHandle,
        window: int = 15,
):
    y = np.array(y)
    yavg = moving_average(y, window=window)
    line[0].set_ydata(y)
    line[0].set_xdata(np.arange(y.shape[0]))
    #  line[0].set_xdata(np.arange(len(yavg)))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)


def update_joint_plots(
        plot_data1: LivePlotData,
        plot_data2: LivePlotData,
        display_id: DisplayHandle,
        window=15,
        alt_loss=None,
):
    x1 = plot_data1.data
    x2 = plot_data2.data
    plot_obj1 = plot_data1.plot_obj
    plot_obj2 = plot_data2.plot_obj

    fig = plt.gcf()

    x1 = np.array(x1).squeeze()
    x2 = np.array(x2).squeeze()
    y1 = moving_average(x1, window=window)
    y2 = moving_average(x2, window=window)
    plot_obj2.line[0].set_ydata(y2)
    plot_obj2.line[0].set_xdata(np.arange(y2.shape[0]))

    plot_obj1.line[0].set_ydata(np.array(y1))
    plot_obj1.line[0].set_xdata(np.arange(y1.shape[0]))
    plot_obj1.ax.relim()
    plot_obj2.ax.relim()
    plot_obj1.ax.autoscale_view()
    plot_obj2.ax.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)  # need to force colab to update plot
