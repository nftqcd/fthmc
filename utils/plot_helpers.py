import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from IPython.display import display

sns.set_palette('bright')


def init_live_plot(n_era, n_epoch, dpi=125, figsize=(8, 4), param=None):
    sns.set_style('ticks')
    fig, ax_ess = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                               constrained_layout=True)
    plt.xlim(0, n_era * n_epoch)
    plt.ylim(0, 1)

    ess_line = ax_ess.plot([0], [0], alpha=0.5, color='C0')  # dummyZ
    _ = ax_ess.set_ylabel('ESS', color='C0')
    _ = ax_ess.tick_params(axis='y', labelcolor='C0')
    _ = ax_ess.grid(False)

    ax_loss = ax_ess.twinx()
    loss_line = ax_loss.plot([0], [0], alpha=0.5, c='C1')  # dummy
    _ = ax_loss.set_ylabel('Loss', color='C1')
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


def update_plots(
        history,
        fig,
        ax_ess,
        ax_loss,
        ess_line,
        loss_line,
        display_id,
        window=15,
):
    y = np.array(history['ess'])
    y = moving_average(y, window=window)
    ess_line[0].set_ydata(y)
    ess_line[0].set_xdata(np.arange(len(y)))
    y = history['loss']
    y = moving_average(y, window=window)
    loss_line[0].set_ydata(np.array(y))
    loss_line[0].set_xdata(np.arange(len(y)))
    ax_loss.relim()
    ax_loss.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)  # need to force colab to update plot
