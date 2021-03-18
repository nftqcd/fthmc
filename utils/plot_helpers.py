import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

    display_id = display(fig, display_id=True)
    return {
        'fig': fig, 'ax': ax, 'line': line, 'display_id': display_id,
    }



def init_live_joint_plots(
        n_era,
        n_epoch,
        dpi=125,
        figsize=(8, 4),
        param=None,
        x_label=None,
        y_label=None,
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
    _ = ax_loss.set_xlabel('Epoch')
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


def update_joint_plots(
        history,
        fig,
        ax_ess,
        ax_loss,
        ess_line,
        loss_line,
        display_id,
        window=15,
        alt_loss=None,
):
    y = np.array(history['ess'])
    y = moving_average(y, window=window)
    ess_line[0].set_ydata(y)
    ess_line[0].set_xdata(np.arange(len(y)))
    if alt_loss is not None:
        y = history[str(alt_loss)]
    else:
        y = history['loss_dkl']

    y = moving_average(y, window=window)
    loss_line[0].set_ydata(np.array(y))
    loss_line[0].set_xdata(np.arange(len(y)))
    ax_loss.relim()
    ax_loss.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)  # need to force colab to update plot
