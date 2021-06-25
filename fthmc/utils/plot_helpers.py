from __future__ import absolute_import, annotations, division, print_function

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import DisplayHandle, display

import fthmc.utils.io as io
from fthmc.config import Param, TrainConfig
from fthmc.utils.logger import in_notebook

MPL_BACKEND = os.environ.get('MPLBACKEND', None)
EXT = (
    'png' if MPL_BACKEND == 'module://itermplot' or in_notebook()
    else 'pdf'
)

logger = io.Logger()

PathLike = Union[str, Path]


@dataclass
class PlotObject:
    #  fig: plt.Figure
    ax: plt.Axes
    line: list[plt.Line2D]


@dataclass
class LivePlotData:
    data: Any
    plot_obj: PlotObject


def torch_delete(x: torch.Tensor, indices: torch.Tensor):
    mask = torch.ones(x.numel(), dtype=torch.bool)
    mask[indices] = False
    return x[mask]


def therm_arr(
        x: Union[np.ndarray, torch.Tensor],
        therm_frac: float = 0.1,
        ret_steps: bool = True
):
    #  taxis = np.argmax(x.shape)
    taxis = 0
    num_steps = x.shape[taxis]
    therm_steps = int(therm_frac * num_steps)
    x = x[therm_steps:]
    t = np.arange(therm_steps, num_steps)
    if ret_steps:
        return x, t
    return x


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def list_to_arr(x: list):
    try:
        return np.array([grab(torch.stack(i)) for i in x])
    except TypeError:
        return np.array([
            grab(i) if isinstance(i, torch.Tensor) else i
            for i in x
        ])


def savefig(
        fig: plt.Figure,
        outfile: PathLike,
        dpi: int = 500,
        verbose: bool = True
):
    io.check_else_make_dir(os.path.dirname(outfile))
    outfile = io.rename_with_timestamp(outfile, fstr='%H%M%S')
    if verbose:
        logger.log(f'Saving figure to: {outfile}')
    #  fig.clf()
    #  plt.close('all')
    if MPL_BACKEND == 'module://itermplot':
        plt.show()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')


def save_live_plots(plots: dict[str, dict], outdir: PathLike):
    if isinstance(outdir, Path):
        outdir = str(outdir)

    io.check_else_make_dir(outdir)
    logger.log(f'Saving live plots to: {outdir}')
    for key, plot in plots.items():
        outfile = os.path.join(outdir, f'{key}_live.{EXT}')
        try:
            savefig(plot['fig'], outfile, verbose=False)
        except KeyError:
            continue


ArrayLike = Union[np.ndarray, torch.Tensor]

def plot_metric(
        metric: Union[ArrayLike, list],
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        hline: bool = False,
        thin: int = 0,
        num_chains: int = 10,
        therm_frac: float = 0.,
        outfile: PathLike = None,
        figsize: tuple = None,
        verbose: bool = False,
        **kwargs,
):
    """Plot metric object."""
    if figsize is None:
        figsize = (4, 3)

    if not isinstance(metric, (list, np.ndarray, torch.Tensor)):
        raise ValueError('metric must be one of '
                         '`list`, `np.ndarray`  or `torch.Tensor`')

    #  if isinstance(metric, list):
    #      if isinstance(metric[0], tuple) and len(metric[0]) == 1:
    #          m = np.
    #          metric = grab(torch.tensor([grab(m[0]) for m in metric]))
    x = metric
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            x = grab(torch.Tensor(torch.stack(x))).squeeze()
        elif isinstance(x[0], np.ndarray):
            x = np.stack(x).squeeze()

    elif isinstance(x, torch.Tensor):
        x = grab(x).squeeze()

    else:
        try:
            x = np.array(x)
        except:
            raise ValueError('Unable to parse metric.')

    #  if isinstance(metric, list) and isinstance(metric[0], torch.Tensor):
    #      metric = grab(torch.Tensor(metric)).squeeze()
    #
    #  if isinstance(metric, torch.Tensor):
    #      metric = grab(torch.Tensor(metric)).squeeze()

    #  metric = np.array(metric)
    if thin > 0:
        x = x[::thin]

    x, steps = therm_arr(x, therm_frac, ret_steps=True)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if len(x.squeeze().shape) == 1:
        label = kwargs.pop('label', None)
        ax.plot(steps, x, label=label, **kwargs)

    if len(x.squeeze().shape) == 2:
        for idx in range(num_chains):
            y = x[:, idx]
            ax.plot(steps, y, **kwargs)

    if hline:
        avg = np.mean(x)
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
        savefig(fig, outfile, verbose=verbose)

    return fig, ax


def plot_history(
        #  history: dict[str, Union[list, torch.Tensor, np.ndarray]],
        history: dict[str, list],
        param: Param,
        therm_frac: float = 0.0,
        config: TrainConfig = None,
        xlabel: str = None,
        title: str = None,
        num_chains: int = 10,
        outdir: str = None,
        skip: list[str] = None,
        thin: int = 0,
        #  hline: bool = False,
        verbose: bool = False,
        **kwargs,
):
    for key, val in history.items():
        if skip is not None and key in skip:
            continue

        if isinstance(val[0], np.ndarray):
            vt = np.stack(val)

        elif isinstance(val[0], torch.Tensor):
            vt = grab(torch.stack(val)).squeeze()
        else:
            try:
                vt = grab(torch.tensor(val, dtype=torch.float)).squeeze()
            except:
                vt = grab(torch.Tensor(torch.stack(val).to(torch.float)))
            finally:
                vt = np.array(val)

        outfile = None
        if outdir is not None:
            outfile = os.path.join(outdir, f'{key}.{EXT}')

        if title is None:
            tarr = get_title(param, config)
            title = '\n'.join(tarr) if len(tarr) > 0 else ''

        _ = plot_metric(vt,
                        ylabel=key,
                        title=title,
                        xlabel=xlabel,
                        outfile=outfile,
                        thin=thin,
                        therm_frac=therm_frac,
                        num_chains=num_chains,
                        verbose=verbose,
                        **kwargs)
    if not in_notebook():
        plt.close('all')
    else:
        plt.show()



yLabel = list[str]
yLabels = list[yLabel]


def init_plots(
        config: TrainConfig = None,
        param: Param = None,
        ylabels: yLabels = None,
        **kwargs,
):
    #  colors_rb = ['#00CCFF', '#f92672']
    #  colors_gp = ['#66ff66', '#ff5fff']
    #  colors_yp = ['#ffff78', '#cc78fa']
    plots_dkl = None
    plots_ess = None
    if in_notebook:
        if ylabels is None:
            ylabels = [['loss_dkl', 'ESS'], ['dq', 'ESS']]

        c0 = ['C0', 'C1']
        plots_dkl = init_live_joint_plots(ylabels=ylabels[0], param=param,
                                          config=config, colors=c0, **kwargs)
        c1 = ['C2', 'C3']
        plots_ess = init_live_joint_plots(ylabels=ylabels[1], param=param,
                                          config=config, colors=c1, **kwargs)

    return {'dkl': plots_dkl, 'ess': plots_ess}


def init_live_joint_plots(
        ylabels: list[str],
        #  n_era: int = None,
        #  n_epoch: int = None,
        dpi: int = 120,
        figsize: tuple = None,
        param: Param = None,
        config: TrainConfig = None,
        xlabel: str = None,
        colors: list[str] = None,
        use_title: bool = False,
        set_xlim: bool = False,
):
    if use_title or set_xlim:
        assert param is not None or config is not None

    if colors is None:
        n = np.random.randint(10, size=2)
        colors = [f'C{n[0]}', f'C{n[1]}']
        #  colors = ['C0', 'C1']
        #  colors = ['#00CCFF', '#f92672']
        #colors = ['#66ff66', '#ff5fff']
        #colors = ['#ffff78', '#cc78fa']

    if figsize is None:
        figsize = (8 ,3)

    fig, ax0 = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                            constrained_layout=True)

    ax1 = ax0.twinx()

    line0 = ax0.plot([0], [0], alpha=0.9, c=colors[0])
    line1 = ax1.plot([0], [0], alpha=0.9, c=colors[1])  # dummy

    ax0.set_ylabel(ylabels[0], color=colors[0])
    ax1.set_ylabel(ylabels[1], color=colors[1])

    ax0.tick_params(axis='y', labelcolor=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[1])

    ax0.grid(False)
    ax1.grid(False)
    ax0.set_xlabel('Epoch' if xlabel is None else xlabel)

    if set_xlim:
        assert config is not None
        plt.xlim(0, config.n_era * config.n_epoch)

    if use_title:
        assert config is not None or param is not None
        title = get_title(param, config)
        if len(title) > 0:
            fig.suptitle('\n'.join(title))

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
        dpi: int = 120,
        figsize: tuple = None,
        param: Param = None,
        config: TrainConfig = None,
        xlabel: str = None,
        ylabel: str = None,
        use_title: bool = True,
        **kwargs
):
    color = kwargs.pop('color', '#87ff00')
    xlabel = 'Epoch' if xlabel is None else xlabel
    if figsize is None:
        figsize = (8, 2.75)
    #  sns.set_style('ticks')
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize, constrained_layout=True)
    line = ax.plot([0], [0], c=color, **kwargs)

    if use_title:
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


def moving_average(x: np.ndarray, window: int = 1):
    #  if len(x) < window:
    if len(x.shape) > 0 and x.shape[0] < window:
        return np.mean(x, keepdims=True)
    #  if len(x.shape) > 0:
    #      avgd = x[-window:].mean(-1)
    #  else:
    #      avgd = x.mean(-1)
    #  if x.shape[0] < window:
    #      return np.mean(x, keepdims=True)

    #  return avgd
    return np.convolve(x, np.ones(window), 'valid') / window


def update_plots(plots, metrics, window: int = 1):
    for key, val in metrics.items():
        if key in plots:
            update_plot(y=val, window=window, **plots[key])


def update_plot(
        y: ArrayLike,
        fig: plt.Figure,
        ax: plt.Axes,
        line: list[plt.Line2D],
        display_id: DisplayHandle,
        window: int = 1,
):
    #  if isinstance(y, torch.Tensor):
    #      if y.requires_grad_:
    #          y = y.detach()
    #      try:
    #          y = y.cpu().numpy()
    #      except AttributeError:
    #          y = y.numpy()
    #  else:
    #      y = np.array(y)

    if isinstance(y, list):
        try:
            y = torch.tensor(y)
        except:
            y = torch.Tensor(torch.stack(y)).detach().cpu().numpy().squeeze()
            #  try:
            #  except:
            #      try:
            #          y = [grab(m[0]) for m in y]
        #  if isinstance(y[0], tuple) and len(y[0]) == 1:
        #      y = [grab(m[0]) for m in y]
        #
        #  if isinstance(y[0], torch.Tensor):
        #      y = torch.Tensor(torch.stack(y)).detach().cpu().numpy().squeeze()
        #      #  y = torch.Tensor(torch.stack(y)).detach().cpu().numpy().squeeze()

    #  if isinstance(y, torch.Tensor):
    #      y = torch.Tensor(y).detach().cpu().numpy().squeeze()

    if isinstance(y, (torch.Tensor, np.ndarray)) and len(y.shape) == 2:
        y = y.mean(-1)

    yavg = moving_average(np.array(y).squeeze(), window=window)

    #  if window > 0 and y.shape[0] > 1:
    #  if window > 0 and len(y.shape) > 0:
    #      if len(y.shape) == 2:
    #          y = y[-window:].mean(-1)
    #      else:
    #          y = y[-window:]
    #      #  y = moving_average(np.array(y).squeeze(), window=window)
    #  avgd = y[-window:]
    #  if window > 0 and len(y.shape) > 0:
    #      if y.shape[0] < window:
    #          window = min((y.shape[0], -1, 1))
    #
    #      avgd = y[-window:].mean(-1)

    line[0].set_ydata(yavg)
    #  line[0].set_xdata(np.arange(y.shape[0]))
    line[0].set_xdata(np.arange(yavg.shape[0]))
    #  line[0].set_xdata(np.arange(len(yavg)))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)


def update_joint_plots(
        plot_data1: LivePlotData,
        plot_data2: LivePlotData,
        display_id: DisplayHandle,
        fig: plt.Figure = None,
        #  ax1: plt.Axes = None,
        #  ax2: plt.Axes = None,
        window=15,
        #  alt_loss=None,
):
    x1 = plot_data1.data
    x2 = plot_data2.data
    plot_obj1 = plot_data1.plot_obj
    plot_obj2 = plot_data2.plot_obj

    if fig is None:
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
