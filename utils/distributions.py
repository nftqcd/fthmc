import torch

from torch.distributions.uniform import Uniform


class MultivariateUniform(torch.nn.Module):
    """Uniformly draw samples from [a, b]."""
    def __init__(self, a, b):
        super().__init__()
        self.dist = Uniform(a, b)

    def log_prob(self, x):
        axes = range(1, len(x.shape))
        return torch.sum(self.dist.log_prob(x), dim=tuple(axes))

    def sample_n(self, batch_size):
        return self.dist.sample((batch_size,))
