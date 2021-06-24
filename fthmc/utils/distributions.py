"""
distributions.py
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.distributions.uniform import Uniform


def bootstrap(x: np.ndarray, *, nboot: int, binsize: int):
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for _ in range(nboot):
        avg = np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0, 1))
        boots.append(avg)

    return np.mean(boots), np.std(boots)


def calc_dkl(logp: torch.Tensor, logq: torch.Tensor):
    return (logq - logp).mean()


def calc_ess(logp: torch.Tensor, logq: torch.Tensor):
    logw = logp - logq
    log_ess = (2 * torch.logsumexp(logw, dim=0)
               - torch.logsumexp(2 * logw, dim=0))

    ess_per_cfg = torch.exp(log_ess)
    if len(logw.shape) > 0:
        ess_per_cfg /= logw.shape[0]

    return ess_per_cfg


class BasePrior(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_n(self, n: int)  -> torch.Tensor:
        raise NotImplementedError


class SimpleNormal(nn.Module):
    def __init__(self, loc: torch.Tensor, var: torch.Tensor):
        self.dist = dist.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


class MultivariateUniform(nn.Module):
    """Uniformly draw samples from [a, b]."""
    def __init__(self, a, b):
        super().__init__()
        self.dist = Uniform(a, b)

    def log_prob(self, x):
        axes = range(1, len(x.shape))
        return torch.sum(self.dist.log_prob(x), dim=tuple(axes))

    def sample_n(self, batch_size):
        return self.dist.sample((batch_size,))
