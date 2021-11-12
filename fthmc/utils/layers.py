"""
layers.py

------ BEGIN CODE FROM https://arxiv.org/abs/2101.08176
Introduction to Normalizing Flows for Lattice Field Theory
Michael S. Albergo, Denis Boyda, Daniel C. Hackett, Gurtej Kanwar, Kyle
Cranmer, Sébastien Racanière, Danilo Jimenez Rezende, Phiala E. Shanahan
License: CC BY 4.0

With slight modifications by Xiao-Yong Jin to reduce global variables
"""
from __future__ import absolute_import, annotations, division, print_function

from math import pi as PI

import numpy as np
import packaging.version
import torch
from torch import nn as nn

from fthmc.config import DEVICE, npDTYPE
from fthmc.utils.logger import Logger
from fthmc.utils.qed_helpers import compute_u1_plaq

#  from torch import nn


TWO_PI = 2 * PI

logger = Logger()

TOL = 1e-6
WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = True
    TOL = 1e-6


# pylint:disable=missing-function-docstring

def torch_mod(x: torch.Tensor):
    #  return torch.remainder(x, TWO_PI)
    return torch.remainder(x + PI, 2 * PI) - PI


def torch_wrap(x: torch.Tensor):
    return torch_mod(x + PI) - PI


def grab(var: torch.Tensor):
    return var.detach().cpu().numpy()


def get_nets(layers: nn.ModuleList) -> list[nn.Module]:
    return [layer.plaq_coupling.net for layer in layers]


def stack_cos_sin(x: torch.Tensor):
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


def tan_transform_symmetric(x: torch.Tensor, s: torch.Tensor):
    return torch_mod(2 * torch.atan(torch.exp(s) * torch.tan(x / 2)))


def tan_transform(x: torch.Tensor, s: torch.Tensor):
    return torch_mod(
        2 * torch.atan(torch.exp(s) * torch.tan(x / 2.))
    )


def tan_transform_logJ(x: torch.Tensor, s: torch.Tensor):
    return -torch.log(
        torch.exp(-s)  * torch.cos(x / 2) ** 2
        + torch.exp(s) * torch.sin(x / 2) ** 2
    )


def mixture_tan_transform(x: torch.Tensor, s: torch.Tensor):
    cond = (len(x.shape) == len(s.shape))
    assert cond, f'Dimension mismatch between x and s: {x.shape}, vs {s.shape}'
    return torch.mean(tan_transform(x, s), dim=1, keepdim=True)


def mixture_tan_transform_logJ(x: torch.Tensor, s: torch.Tensor):
    cond = (len(x.shape) == len(s.shape))
    assert cond, f'Dimension mismatch between x and s: {x.shape}, vs {s.shape}'
    return (
        torch.logsumexp(tan_transform_logJ(x, s), dim=1) - np.log(s.shape[1])
    )


def make_net_from_layers(
        *,
        lattice_shape: tuple,
        nets: list[nn.Module]
):
    n_layers = len(nets)
    layers = []
    for i in range(n_layers):
        # periodically loop through all arrangements of maskings
        mu = i % 2
        off = (i // 2) % 4
        net = nets[i]
        plaq_coupling = NCPPlaqCouplingLayer(
            net, mask_shape=lattice_shape, mask_mu=mu, mask_off=off
        )
        link_coupling = GaugeEquivCouplingLayer(
            lattice_shape=lattice_shape, mask_mu=mu, mask_off=off,
            plaq_coupling=plaq_coupling
        )
        layers.append(link_coupling)

    return nn.ModuleList(layers)


ACTIVATION_FNS = {
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
    'swish': nn.SiLU(),
    'leaky_relu': nn.LeakyReLU(),
}

def get_activation_fn(actfn: str = None):
    if actfn is None:
        return nn.SiLU()

    actfn = ACTIVATION_FNS.get(str(actfn).lower(), None)
    if actfn is None:
        logger.log(f'Activation fn: {actfn} not found. '
                   f'Must be one of ["relu", "silu", "leaky_relu"]')
        logger.log(f'Falling back to: nn.SiLU()')
        return nn.SiLU()

    return actfn


def make_conv_net(
        *,
        hidden_sizes: list[int],
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        use_final_tanh: bool = False,
        activation_fn: str = None,
):
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch>=1.5.0'
    assert (packaging.version.parse(torch.__version__)
            >= packaging.version.parse('1.5.0'))

    net = []
    padding_size = (kernel_size // 2)
    act_fn = get_activation_fn(actfn=activation_fn)
    # hidden_sizes = [8, 8]
    sizes = [in_channels] + hidden_sizes + [out_channels]
    for i in range(len(sizes) - 1):
        net.append(nn.Conv2d(sizes[i], sizes[i+1],
                             kernel_size, stride=1,
                             padding=padding_size,
                             padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(act_fn)
        else:
            if use_final_tanh:
                net.append(nn.Tanh())

    return nn.Sequential(*net)


def set_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight, mean=1, std=2)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.fill_(-1)


def gauge_transform(links, alpha):
    for mu in range(len(links.shape[2:])):
        links[:,mu] = alpha + links[:,mu] - torch.roll(alpha, -1, mu+1)
    return links


def random_gauge_transform(x):
    Nconf, VolShape = x.shape[0], x.shape[2:]
    return gauge_transform(x, TWO_PI * torch.rand((Nconf,) + VolShape))


class GaugeEquivCouplingLayer(nn.Module):
    """U(1) gauge equiv coupling layer defined by `plaq_coupling` acting on plaquettes."""
    def __init__(self, *, lattice_shape, mask_mu, mask_off, plaq_coupling):
        super().__init__()
        link_mask_shape = (len(lattice_shape),) + lattice_shape
        self.active_mask = make_2d_link_active_stripes(link_mask_shape, mask_mu, mask_off)
        self.plaq_coupling = plaq_coupling

    def forward(self, x):
        plaq = compute_u1_plaq(x, mu=0, nu=1)
        new_plaq, logJ = self.plaq_coupling(plaq)
        delta_plaq = new_plaq - plaq
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1) # signs for U vs Udagger
        fx = self.active_mask * torch_mod(delta_links + x) + (1-self.active_mask) * x
        return fx, logJ

    def reverse(self, fx):
        new_plaq = compute_u1_plaq(fx, mu=0, nu=1)
        plaq, logJ = self.plaq_coupling.reverse(new_plaq)
        delta_plaq = plaq - new_plaq
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1) # signs for U vs Udagger
        x = self.active_mask * torch_mod(delta_links + fx) + (1-self.active_mask) * fx
        return x, logJ


def make_2d_link_active_stripes(shape, mu, off):
    """
    Stripes mask looks like in the `mu` channel (mu-oriented links)::

      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction, and the pattern is offset in the nu
    direction by `off` (mod 4). The other channel is identically 0.
    """
    assert len(shape) == 2+1, 'need to pass shape suitable for 2D gauge theory'
    assert shape[0] == len(shape[1:]), 'first dim of shape must be Nd'
    assert mu in (0,1), 'mu must be 0 or 1'

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[mu, :, 0::4] = 1
    elif mu == 1:
        mask[mu, 0::4] = 1

    nu = 1 - mu
    mask = np.roll(mask, off, axis=nu+1)
    return torch.from_numpy(mask.astype(npDTYPE)).to(DEVICE)


def make_single_stripes(shape, mu, off):
    """
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction. Vector of 1 is repeated every 4.
    The pattern is offset in perpendicular to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, 'need to pass 2D shape'
    assert mu in (0,1), 'mu must be 0 or 1'

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:,0::4] = 1
    elif mu == 1:
        mask[0::4] = 1
    mask = np.roll(mask, off, axis=1-mu)
    return torch.from_numpy(mask).to(DEVICE)

def make_double_stripes(shape, mu, off):
    """
    Double stripes mask looks like::

      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0

    where vertical is the `mu` direction. The pattern is offset in perpendicular
    to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, 'need to pass 2D shape'
    assert mu in (0,1), 'mu must be 0 or 1'

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:,0::4] = 1
        mask[:,1::4] = 1
    elif mu == 1:
        mask[0::4] = 1
        mask[1::4] = 1
    mask = np.roll(mask, off, axis=1-mu)
    return torch.from_numpy(mask).to(DEVICE)


def make_plaq_masks(mask_shape, mask_mu, mask_off):
    mask = {}
    mask['frozen'] = make_double_stripes(mask_shape, mask_mu, mask_off+1)
    mask['active'] = make_single_stripes(mask_shape, mask_mu, mask_off)
    mask['passive'] = 1 - mask['frozen'] - mask['active']
    return mask

def invert_transform_bisect(y, *, f, tol, max_iter, a=-PI, b=PI):
    min_x = a*torch.ones_like(y)
    max_x = b*torch.ones_like(y)
    min_val = f(min_x)
    max_val = f(max_x)
    with torch.no_grad():
        for i in range(max_iter):
            mid_x = (min_x + max_x) / 2
            mid_val = f(mid_x)
            greater_mask = (y > mid_val).int()
            greater_mask = greater_mask.float()
            err = torch.max(torch.abs(y - mid_val))
            if err < tol: return mid_x
            #  if err < TOL: return mid_x
            if torch.all((mid_x == min_x) + (mid_x == max_x)):
                print('WARNING: Reached floating point '
                      f'precision before tolerance (iter {i}, err {err})')
                return mid_x
            min_x = greater_mask*mid_x + (1-greater_mask)*min_x
            min_val = greater_mask*mid_val + (1-greater_mask)*min_val
            max_x = (1-greater_mask)*mid_x + greater_mask*max_x
            max_val = (1-greater_mask)*mid_val + greater_mask*max_val
        print(
            f'WARNING: Did not converge to tol {TOL} in {max_iter} iters! '
            f'Error was {err}'
        )
        return mid_x


# pylint:disable=invalid-name
class NCPPlaqCouplingLayer(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        *,
        mask_shape: tuple[int],
        mask_mu: int,
        mask_off: int,
        inv_prec: float = TOL,
        inv_max_iter: int = 1000
    ):
        super().__init__()
        assert len(mask_shape) == 2, (
            f'NCPPlaqCouplingLayer is implemented only in 2D, '
            f'mask shape {mask_shape} is invalid')
        self.mask = make_plaq_masks(mask_shape, mask_mu, mask_off)
        self.net = net
        self.inv_prec = inv_prec
        self.inv_max_iter = inv_max_iter
        if WITH_CUDA:
            self.net = self.net.cuda()
            for _, val in self.mask.items():
                val = val.cuda()

    def forward(self, x: torch.Tensor):
        if WITH_CUDA:
            x = x.cuda()

        x2 = self.mask['frozen'] * x
        net_out = self.net(stack_cos_sin(x2))
        cond = net_out.shape[1] >= 2
        assert cond, 'CNN must output n_mix (s_i) + 1 (t) channels'

        s, t = net_out[:,:-1], net_out[:,-1]

        x1 = self.mask['active'] * x
        x1 = x1.unsqueeze(1)
        local_logJ = self.mask['active'] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)
        fx1 = self.mask['active'] * mixture_tan_transform(x1, s).squeeze(1)

        fx = (
            self.mask['active'] * torch_mod(fx1 + t)
            + self.mask['passive'] * x
            + self.mask['frozen'] * x
        )
        return fx, logJ

    def reverse(self, fx: torch.Tensor):
        fx2 = self.mask['frozen'] * fx
        net_out = self.net(stack_cos_sin(fx2))
        assert net_out.shape[1] >= 2, (
            'CNN must output n_mix (s_i) + 1 (t) channels'
        )
        s, t = net_out[:,:-1], net_out[:,-1]

        x1 = torch_mod(self.mask['active'] * (fx - t).unsqueeze(1))
        transform = lambda x: self.mask['active'] * mixture_tan_transform(x, s)
        x1 = invert_transform_bisect(
            x1, f=transform, tol=self.inv_prec, max_iter=self.inv_max_iter
        )
        local_logJ = self.mask['active'] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = -torch.sum(local_logJ, dim=axes)
        x1 = x1.squeeze(1)

        x = (
            self.mask['active'] * x1
            + self.mask['passive'] * fx
            + self.mask['frozen'] * fx2
        )
        return x, logJ


def make_u1_equiv_layers(
        *,
        n_layers,
        n_mixture_comps,
        lattice_shape,
        hidden_sizes,
        kernel_size,
        activation_fn: str = None,
):
    layers = []
    for i in range(n_layers):  # ~ 16 layers
        # periodically loop through all arrangements of maskings
        mu = i % 2
        off = (i//2) % 4
        in_channels = 2 # x - > (cos(x), sin(x))
        out_channels = n_mixture_comps + 1 # for mixture s and t, respectively
        net = make_conv_net(in_channels=in_channels,
                            out_channels=out_channels,
                            hidden_sizes=hidden_sizes,
                            kernel_size=kernel_size,
                            use_final_tanh=False,
                            activation_fn=activation_fn)
        plaq_coupling = NCPPlaqCouplingLayer(net,
                                             mask_shape=lattice_shape,
                                             mask_mu=mu, mask_off=off)
        link_coupling = GaugeEquivCouplingLayer(lattice_shape=lattice_shape,
                                                mask_mu=mu, mask_off=off,
                                                plaq_coupling=plaq_coupling)
        layers.append(link_coupling)

    return nn.ModuleList(layers)
