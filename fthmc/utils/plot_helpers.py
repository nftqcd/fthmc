from __future__ import absolute_import, division, print_function, annotations
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fthmc.utils.param import Param
import fthmc.utils.io as io

from fthmc.config import PlotObject, LivePlotData, TrainConfig

from IPython.display import display

logger = io.Logger()


def therm_arr(
        x: np.ndarray,
        therm_frac: float = 0.1,
        ret_steps: bool = True
):
    taxis = np.argmax(x.shape)
    num_steps = x.shape[taxis]
    therm_steps = int(therm_frac * num_steps)
    x = np.delete(x, np.s_[:therm_steps], axis=taxis)
    t = np.arange(therm_steps, num_steps)
    if ret_steps:
        return x, t
    return x

def savefig(fig: plt.Figure, outfile: str):
    io.check_else_make_dir(os.path.dirname(outfile))
    logger.log(f'Saving figure to: {outfile}')
    fig.savefig(outfile, dpi=500, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def plot_metric(
        metric: np.ndarray,
        therm_frac: float = 0.,
        outfile: str = None,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        figsize: tuple[int] = None,
        num_chains: int = 10,
        **kwargs,
):
    """Plot metric object."""
    if figsize is None:
        figsize = (4, 3)

    metric = np.array(metric)
    metric, steps = therm_arr(metric, therm_frac, ret_steps=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if len(metric.shape) == 1:
        ax.plot(steps, metric, **kwargs)

    if len(metric.shape) == 2:
        for idx in range(num_chains):
            y = metric[:, idx]
            ax.plot(steps, y, **kwargs)

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
        param: Param,
        history: dict[[str], np.ndarray],
        therm_frac: float = 0.0,
        xlabel: str = None,
        title: str = None,
        num_chains: int = 10,
        outdir: str = None,
):
    for key, val in history.items():
        outfile = None
        if outdir is not None:
            outfile = os.path.join(outdir, f'{key}.pdf')

        if title is None:
            title = param.uniquestr()

        _ = plot_metric(val,
                        ylabel=key,
                        title=title,
                        xlabel=xlabel,
                        outfile=outfile,
                        therm_frac=therm_frac,
                        num_chains=num_chains)


def init_plots(config: TrainConfig, param: Param, figsize: tuple = (8, 3)):
    plots_dq = {}
    plots_dkl = {}
    plots_force = {}
    if io.in_notebook():
        ylabel_dq = ['dq', 'ESS']
        plots_dq = init_live_joint_plots(config.n_era, config.n_epoch,
                                         dpi=500, figsize=figsize, param=param,
                                         ylabel=ylabel_dq)
        ylabel_dkl = ['loss_dkl', 'ESS']
        plots_dkl = init_live_joint_plots(config.n_era, config.n_epoch,
                                          dpi=500, figsize=figsize, param=param,
                                          ylabel=ylabel_dkl)

        if config.with_force:
            ylabel_force = ['loss_force', 'ESS']
            plots_force = init_live_joint_plots(config.n_era,
                                                config.n_epoch, dpi=500,
                                                figsize=figsize,
                                                param=param,
                                                ylabel=ylabel_force)

    return {
        'dq': plots_dq,
        'dkl': plots_dkl,
        'force': plots_force,
    }


def init_live_joint_plots(
    n_era: int,
    n_epoch: int,
    dpi: int = 400,
    figsize: tuple = (8, 4),
    param: Param = None,
    xlabel: str = None,
    ylabel: list[str] = None,
    colors: list[str] = None,
):
    if colors is None:
        colors = ['C0', 'C1']

    #  sns.set_style('ticks')
    fig, ax0 = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                            constrained_layout=True)
    plt.xlim(0, n_era * n_epoch)
    line0 = ax0.plot([0], [0], alpha=0.5, color='C0')
    ax1 = ax0.twinx()
    if ylabel is None:
        ax0.set_ylabel('Loss', color=colors[0])
        ax1.set_ylabel('ess', color=colors[1])

    else:
        ax0.set_ylabel(ylabel[0], color=colors[0])
        ax1.set_ylabel(ylabel[1], color=colors[1])

    ax0.tick_params(axis='y', labelcolor=colors[0])
    ax0.grid(False)

    line1 = ax1.plot([0], [0], alpha=0.5, c=colors[1])  # dummy

    ax0.tick_params(axis='y', labelcolor=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[1])
    ax1.grid(False)
    ax0.set_xlabel('Epoch' if xlabel is None else xlabel)
    if param is not None:
        fig.suptitle(param.uniquestr())

    display_id = display(fig, display_id=True)
    plot_obj1 = PlotObject(ax0, line0)
    plot_obj2 = PlotObject(ax1, line1)
    return {
        'plot_obj1': plot_obj1,
        'plot_obj2': plot_obj2,
        'display_id': display_id
    }


def init_live_plot(
        dpi=125,
        figsize=(8, 4),
        param=None,
        x_label=None,
        y_label=None,
        **kwargs
):
    #  sns.set_style('ticks')
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize, constrained_layout=True)
    line = ax.plot([0], [0], **kwargs)
    if param is not None:
        _ = fig.suptitle(param.uniquestr())

    if x_label is not None:
        _ = ax.set_xlabel(x_label)

    if y_label is not None:
        _ = ax.set_ylabel(y_label)

    _ = ax.autoscale(True, axis='y')
    #  plt.Axes.autoscale(True, axis='y')
    display_id = display(fig, display_id=True)
    return {
        'fig': fig, 'ax': ax, 'line': line, 'display_id': display_id,
    }


def init_live_joint_plots1(
        n_era: int,
        n_epoch: int,
        dpi: int = 125,
        figsize: tuple = (8, 4),
        param: Param = None,
        x_label: str = None,
        y_label: str = None,
):
    #  sns.set_style('ticks')
    fig, ax_ess = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                               constrained_layout=True)
    plt.xlim(0, n_era * n_epoch)
    plt.ylim(0, 1)

    ess_line = ax_ess.plot([0], [0], alpha=0.5, color='C0')  # dummyZ
    if y_label is None:
        _ = ax_ess.set_ylabel('ESS', color='C0')
    else:
        _ = ax_ess.set_ylabel(y_label[0], color='C0')

    _ = ax_ess.tick_params(axis='y', labelcolor='C0')
    _ = ax_ess.grid(False)

    ax_loss = ax_ess.twinx()
    loss_line = ax_loss.plot([0], [0], alpha=0.5, c='C1')  # dummy
    if y_label is None:
        _ = ax_loss.set_ylabel('Loss', color='C1')
    else:
        _ = ax_loss.set_ylabel(y_label[1], color='C1')

    _ = ax_loss.tick_params(axis='y', labelcolor='C1')
    _ = ax_loss.grid(False)
    _ = ax_loss.set_xlabel('Epoch' if x_label is None else x_label)
    if param is not None:
        _ = fig.suptitle(param.uniquestr())

    display_id = display(fig, display_id=True)
    return {
        'fig': fig,
        'ax_ess': ax_ess,
        'ax_loss': ax_loss,
        'ess_line': ess_line,
        'loss_line': loss_line,
        'display_id': display_id
    }


def moving_average(x: np.ndarray, window: int = 10):
    if len(x) < window:
        return np.mean(x, keepdims=True)

    return np.convolve(x, np.ones(window), 'valid') / window


def update_plot(
        y: np.ndarray,
        fig: plt.Figure,
        ax: plt.Axes,
        line: list[plt.Line2D],
        display_id: int,
        window: int = 15,
):
    y = np.array(y)
    y = moving_average(y, window=window)
    line[0].set_ydata(y)
    line[0].set_xdata(np.arange(len(y)))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)


def update_joint_plots(
        plot_data1: LivePlotData,
        plot_data2: LivePlotData,
        display_id,
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
