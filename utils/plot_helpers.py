import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .param import Param
from dataclasses import dataclass

from typing import Any

from IPython.display import display

sns.set_palette('bright')


def init_live_plot(
        n_era,
        n_epoch,
        dpi=125,
        figsize=(8, 4),
        param=None,
        x_label=None,
        y_label=None,
        **kwargs
):
    sns.set_style('ticks')
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


def init_live_joint_plots(
    n_era: int,
    n_epoch: int,
    dpi: int = 400,
    figsize: tuple = (8, 4),
    param: Param = None,
    x_label: str = None,
    y_label: list = None,
):
    sns.set_style('ticks')
    fig, ax0 = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                            constrained_layout=True)
    plt.xlim(0, n_era * n_epoch)
    line0 = ax0.plot([0], [0], alpha=0.5, color='C0')
    ax1 = ax0.twinx()
    if y_label is None:
        _ = ax0.set_ylabel('Loss', color='C0')
        _ = ax1.set_ylabel('ess', color='C1')

    else:
        _ = ax0.set_ylabel(y_label[0], color='C0')
        _ = ax1.set_ylabel(y_label[1], color='C1')

    _ = ax0.tick_params(axis='y', labelcolor='C0')
    _ = ax0.grid(False)

    line1 = ax1.plot([0], [0], alpha=0.5, c='C1')  # dummy
    if y_label is None:
        _ = ax1.set_ylabel('ESS', color='C1')
    else:
        _ = ax1.set_ylabel(y_label[1], color='C1')

    _ = ax0.tick_params(axis='y', labelcolor='C0')
    _ = ax1.tick_params(axis='y', labelcolor='C1')
    _ = ax1.grid(False)
    _ = ax0.set_xlabel('Epoch' if x_label is None else x_label)
    if param is not None:
        _ = fig.suptitle(param.uniquestr())

    display_id = display(fig, display_id=True)
    plot_obj1 = PlotObject(ax0, line0)
    plot_obj2 = PlotObject(ax1, line1)
    return {
        'plot_obj1': plot_obj1,
        'plot_obj2': plot_obj2,
        'display_id': display_id
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
    sns.set_style('ticks')
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


def moving_average(x, window=10):
    if len(x) < window:
        return np.mean(x, keepdims=True)

    return np.convolve(x, np.ones(window), 'valid') / window


def update_plot(
        y,
        fig,
        ax,
        line,
        display_id,
        window=15,
):
    y = np.array(y)
    y = moving_average(y, window=window)
    line[0].set_ydata(y)
    line[0].set_xdata(np.arange(len(y)))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)


@dataclass
class PlotObject:
    ax: plt.Axes
    line: plt.Line2D


@dataclass
class LivePlotData:
    data: Any
    plot_obj: PlotObject


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
    y1 = moving_average(np.array(x1), window=window)
    y2 = moving_average(np.array(x2), window=window)

    #  y = moving_average(y, window=window)
    plot_obj2.line[0].set_ydata(y2)
    plot_obj2.line[0].set_xdata(np.arange(y2.shape[0]))

    plot_obj1.line[0].set_ydata(np.array(y1))
    plot_obj1.line[0].set_xdata(np.arange(y1.shape[0]))
    plot_obj1.ax.relim()
    plot_obj2.ax.relim()
    plot_obj1.ax.autoscale_view()
    plot_obj2.ax.autoscale_view()
    fig.canvas.draw()
    #  ess_line[0].set_ydata(y)
    #  ess_line[0].set_xdata(np.arange(len(y)))
    #  if alt_loss is not None:
    #      y = history[str(alt_loss)]
    #  else:
    #      y = history['loss']
    #
    #  y = moving_average(y, window=window)
    #  loss_line[0].set_ydata(np.array(y))
    #  loss_line[0].set_xdata(np.arange(len(y)))
    #  ax_loss.relim()
    #  ax_loss.autoscale_view()
    #  fig.canvas.draw()
    display_id.update(fig)  # need to force colab to update plot
