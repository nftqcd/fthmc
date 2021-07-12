"""
inference.py
"""
from __future__ import absolute_import, print_function, division, annotations
from fthmc.config import DTYPE, FlowModel
import torch
from torch.utils.tensorboard.writer import SummaryWriter


def compute_u1_plaq(x, mu, nu):
    """Compute U(1) plaqs in the (mu, nu) plane given `links = arg(U)"""
    return (x[:, mu]
            + torch.roll(x[:, nu], -1, mu + 1)
            - torch.roll(x[:, mu], -1, nu + 1)
            - x[:, nu])


class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x: torch.Tensor):
        nd = x.shape[1]
        action_density = 0
        for mu in range(nd):
            for nu in range(mu + 1, nd):
                action_density = action_density + torch.cos(
                    compute_u1_plaq(x, mu, nu)
                )

        action = torch.sum(action_density, dim=tuple(range(1, nd+1)))
        return - self.beta * action


def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq = logq - logJ

    return x, logq


def serial_sample_generator(model, action, batch_size, num_samples):
    try:
        layers, prior = model.layers, model.prior
    except AttributeError:
        layers, prior = model['layers'], model['prior']
    #  if isinstance(model, dict):
    #  elif isinstance(model, FlowModel):
    #      layers, prior = model.layers, model.prior

    layers.eval()
    x, logq, logp = None, None, None
    for i in range(num_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
            logp = -action(x)

        yield x[batch_i], logq[batch_i], logp[batch_i]


def make_mcmc_ensemble(
    model,
    action_fn,
    batch_size,
    num_samples,
    writer: SummaryWriter = None
):
    history = {
        'x': [],
        'logq': [],
        'logp': [],
        'accepted': [],
    }

    sample_gen = serial_sample_generator(model, action_fn,
                                         batch_size, num_samples)
    step = 0
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:
            accepted = True
        else:
            # Metropolis acceptance condition
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp(
                (new_logp - new_logq) - (last_logp - last_logq)
            )
            p_accept = min(1, p_accept)
            draw = torch.rand(1)
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logq = last_logq
                new_logp = last_logp

        # Update Markov chain
        metrics = {
            'logp': new_logp,
            'logq': new_logq,
            'x': new_x,
            'accepted': accepted,
        }
        for key, val in metrics.items():
            try:
                history[key].append(val)
            except KeyError:
                history[key] = [val]

            val = torch.tensor(val, dtype=DTYPE)
            if writer is not None:
                if len(val.shape) > 1:
                    writer.add_histogram(f'inference/{key}', val,
                                         global_step=step)
                else:
                    writer.add_scalar(f'inference/{key}', val.mean(),
                                      global_step=step)

        step += 1

        #  history['logp'].append(new_logp)
        #  history['logq'].append(new_logq)
        #  history['x'].append(new_x)
        #  history['accepted'].append(accepted)

    return history


