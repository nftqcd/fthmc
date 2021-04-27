import torch
import numpy as np


def apply_flow_to_prior(prior, coupling_layers, *, batch_size, xi=None):
    if xi is None:
        xi = prior.sample_n(batch_size)

    x = xi
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logdet = layer.forward(x)
        logq = logq - logdet

    return xi, x, logq



def serial_sample_generator(model, action_fn, batch_size, num_samples):
    layers = model['layers']
    prior = model['prior']
    layers.eval()
    x, logq, logp = None, None, None
    for i in range(num_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            xi, x, logq = apply_flow_to_prior(prior, layers,
                                              batch_size=batch_size)
            logp = -action_fn(x)

        yield x[batch_i], logq[batch_i], logp[batch_i]



def make_mcmc_ensemble(model, action_fn, batch_size, num_samples):
    names = ['x', 'logq', 'logp', 'accepted']
    history = {
        name: [] for name in names
    }

    # Build Markov chain
    sample_gen = serial_sample_generator(model, action_fn, batch_size,
                                         num_samples)
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:  # always accept the first proposal
            accepted = True
        else:
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp(
                (new_logp - new_logq) - (last_logp - last_logq)
            )
            p_accept = min(1, p_accept)
            draw = torch.rand(1)  # ~ [0, 1]
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logp = last_logp
                new_logq = last_logq

        # Update Markov Chain
        history['logp'].append(new_logp)
        history['logq'].append(new_logq)
        history['x'].append(new_x)
        history['accepted'].append(accepted)

    return history
