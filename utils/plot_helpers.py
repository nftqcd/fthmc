import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display


def init_live_plot(n_era, n_epoch, dpi=125, figsize=(8, 4)):
    fig, ax_ess = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    plt.xlim(0, n_era * n_epoch)
    plt.ylim(0, 1)

    ess_line = plt.plot([0], [0], alpha=0.5)  # dummy
    plt.grid(False)
    plt.ylabel('ESS')

    ax_loss = ax_ess.twinx()
    loss_line = plt.plot([0], [0], alpha=0.5, c='orange')  # dummy
    plt.grid(False)
    plt.ylabel('Loss')

    plt.xlabel('Epoch')
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
