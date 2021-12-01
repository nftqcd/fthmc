"""
inference.py
"""
from __future__ import absolute_import, print_function, division, annotations
from dataclasses import dataclass, field
from fthmc.utils.samplers import ActionFn
from fthmc.config import BaseHistory, DTYPE, FlowModel
import torch
from torch.utils.tensorboard.writer import SummaryWriter


def drop_nans_from_tensor(x: torch.Tensor):
    # flatten and remember original shape
    shape = x.shape
    y = x.reshape(shape[0], -1)
    # Drop all rows containing any nan
    y = y[~torch.any(y.isnan(), dim=1)]
    # reshape back
    return y.reshape(y.shape[0], *shape[1:])




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

    layers.eval()
    x, logq, logp = None, None, None
    for i in range(num_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
            logp = -action(x)

        yield x[batch_i], logq[batch_i], logp[batch_i]


def update_summaries(
        step: int,
        writer: SummaryWriter,
        metrics: dict[str, torch.Tensor],
        pre: str = None
):
    for key, val in metrics.items():
        if pre is not None:
            key = '/'.join([pre, key])

        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, dtype=DTYPE)

        writer.add_scalar(key, val.mean(), global_step=step)
        if len(val.shape) > 1:
            if torch.any(val.isnan()):
                v = drop_nans_from_tensor(val)
                if v.shape[0] == 0:
                    continue

            writer.add_histogram(key, val, global_step=step)



@dataclass
class History(BaseHistory):
    """The list of variables to be tracked inside `make_mcmc_ensemble`."""
    x: list[torch.Tensor] = field(default_factory=list)
    logp: list[torch.Tensor] = field(default_factory=list)
    logq: list[torch.Tensor] = field(default_factory=list)
    accepted: list[torch.Tensor] = field(default_factory=list)


def make_mcmc_ensemble(
        model: FlowModel,
        action_fn: ActionFn,
        batch_size: int,
        num_samples: int,
        writer: SummaryWriter = None
):
    history = History()

    sample_gen = serial_sample_generator(model, action_fn,
                                         batch_size, num_samples)
    step = 0
    for new_x, new_logq, new_logp in sample_gen:
        #  if len(history['logp']) == 0:
        if len(history.logp) == 0:
            accepted = True
        else:
            # Metropolis acceptance condition
            #  last_logp = history['logp'][-1]
            #  last_logq = history['logq'][-1]
            last_logp = history.logp[-1]
            last_logq = history.logq[-1]
            p_accept = torch.exp(
                (new_logp - new_logq) - (last_logp - last_logq)
            )
            p_accept = min(1, p_accept)
            draw = torch.rand(1)
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history.x[-1]
                new_logq = last_logq
                new_logp = last_logp

        # Update Markov chain
        metrics = {
            'logp': new_logp,
            'logq': new_logq,
            'x': new_x,
            'accepted': accepted,
        }
        history.update(metrics)
        if writer is not None:
            update_summaries(step, writer, metrics, pre='inference')

        step += 1

        #  for key, val in metrics.items():
        #      try:
        #          history[key].append(val)
        #      except KeyError:
        #          history[key] = [val]
        #      val = torch.tensor(val, dtype=DTYPE)
        #      if writer is not None:
        #          if len(val.shape) > 1:
        #              writer.add_histogram(f'inference/{key}', val,
        #                                   global_step=step)
        #          else:
        #              writer.add_scalar(f'inference/{key}', val.mean(),
        #                                global_step=step)
        #

        #  history['logp'].append(new_logp)
        #  history['logq'].append(new_logq)
        #  history['x'].append(new_x)
        #  history['accepted'].append(accepted)

    return history
