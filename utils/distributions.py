import torch
import numpy as np

from torch.distributions.uniform import Uniform

def bootstrap(x: np.ndarray, *, nboot: int, binsize: int):
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for _ in range(nboot):
        avg = np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0, 1))
        boots.append(avg)

    #  boots = np.array([
    #      np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0, 1))
    #      for _ in range(nboot)
    #  ])

    return np.mean(boots), np.std(boots)


def calc_dkl(logp, logq):
    return (logq - logp).mean()


def calc_ess(logp, logq):
    logw = logp - logq
    log_ess = (2 * torch.logsumexp(logw, dim=0)
               - torch.logsumexp(2 * logw, dim=0))
    ess_per_cfg = torch.exp(log_ess) / len(logw)

    return ess_per_cfg


class SimpleNormal(torch.nn.Module):
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc),
                                                      torch.flatten(var))
        self.shape = loc.shape

    def log_prob(self, x, beta=1.):
        logp = beta * self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


class MultivariateUniform(torch.nn.Module):
    """Uniformly draw samples from [a, b]."""
    def __init__(self, a, b):
        super().__init__()
        self.dist = Uniform(a, b)

    def log_prob(self, x, beta=1.):
        axes = range(1, len(x.shape))
        return torch.sum(beta * self.dist.log_prob(x), dim=tuple(axes))

    def sample_n(self, batch_size):
        return self.dist.sample((batch_size,))
