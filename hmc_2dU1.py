# Copyright (c) 2021 Xiao-Yong Jin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import sys
import math

from timeit import default_timer as timer
from functools import reduce

import numpy as np
import packaging.version

import torch

PI = math.pi
TWO_PI = 2 * PI


put = lambda s: sys.stdout.write(s)

torch_device = 'cpu'
float_dtype = np.float64

class Param:
    def __init__(
        self,
        beta: float = 6.0,
        lat: list = [64, 64],
        tau: float = 2.0,
        nstep: int = 50,
        ntraj: int = 256,
        nrun: int = 4,
        nprint: int = 256,
        seed: int = 11*13,
        randinit: bool = False,
        nth: int = int(os.environ.get('OMP_NUM_THREADS', '2')),
        nth_interop: int = 2
    ):
        self.beta = beta
        self.lat = lat
        self.nd = len(lat)
        self.volume = reduce(lambda x, y: x * y, lat)
        self.tau = tau
        self.nstep = nstep
        self.dt = self.tau / self.nstep
        self.ntraj = ntraj
        self.nrun = nrun
        self.nprint = nprint
        self.seed = seed
        self.randinit = randinit
        self.nth = nth
        self.nth_interop = nth_interop

    def initializer(self):
        if self.randinit:
            return torch.empty((param.nd,) + param.lat).uniform_(-PI, PI)
        else:
            return torch.zeros((param.nd,) + param.lat)

    def summary(self):
        status = {
            'latsize': self.lat,
            'volume': self.volume,
            'beta': self.beta,
            'trajs': self.ntraj,
            'tau': self.tau,
            'steps': self.nstep,
            'seed': self.seed,
            'nth': self.nth,
            'nth_interop': self.nth_interop,
        }

        return ', '.join('='.join((str(k), str(v))) for k, v in status.items())

    def uniquestr(self):
        lat = ".".join(str(x) for x in self.lat)
        return (
            f'out_l{lat}_b{param.beta}_n{param.ntraj}'
            f'_t{param.tau}_s{param.nstep}.out'
        )


def action(param, f):
    return (-param.beta)*torch.sum(torch.cos(plaqphase(f)))


def force(param, f):
    f.requires_grad_(True)
    s = action(param, f)
    s.backward()
    ff = f.grad
    f.requires_grad_(False)

    return ff


def plaqphase(f):
    return (
        f[0, :]
        - f[1, :]
        - torch.roll(f[0, :], shifts=-1, dims=1)
        + torch.roll(f[1, :], shifts=-1, dims=0)
    )


def topocharge(f):
    return torch.floor(0.1 + torch.sum(regularize(plaqphase(f))) / (2 * PI))


def regularize(f):
    f_ = (f - PI) / TWO_PI
    return TWO_PI * (f_ - torch.floor(f_) - 0.5)


def leapfrog(param, x, p):
    dt = param.dt
    x_ = x + 0.5 * dt * p
    p_ = p + (-dt) * force(param, x_)
    for i in range(param.nstep-1):
        x_ = x_ + dt * p_
        p_ = p_ + (-dt) * force(param, x_)
    x_ = x_ + 0.5 * dt * p_

    return (x_, p_)


def hmc(param, x: torch.Tensor):
    p = torch.randn_like(x)
    act0 = action(param, x) + 0.5 * torch.sum(p*p)
    x_, p_ = leapfrog(param, x, p)
    xr = regularize(x_)
    act = action(param, xr) + 0.5 * torch.sum(p_*p_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act - act0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    newx = xr if acc else x
    return (dH, exp_mdH, acc, newx)


# ------ BEGIN CODE FROM https://arxiv.org/abs/2101.08176
# "Introduction to Normalizing Flows for Lattice Field Theory"
#  Michael S. Albergo, Denis Boyda, Daniel C. Hackett, Gurtej Kanwar, Kyle
#  Cranmer, Sébastien Racanière, Danilo Jimenez Rezende, Phiala E. Shanahan
#  License: CC BY 4.0
#  With slight modifications by Xiao-Yong Jin to reduce global variables
def torch_mod(x):
    return torch.remainder(x, 2*np.pi)


def torch_wrap(x):
    return torch_mod(x+np.pi) - np.pi


def grab(var):
    return var.detach().cpu().numpy()


def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)

    return ess_per_cfg

def bootstrap(x, *, Nboot, binsize):
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0,1)))

    return np.mean(boots), np.std(boots)


def print_metrics(history, avg_last_N_epochs, era, epoch):
    print(f'== Era {era} | Epoch {epoch} metrics ==')
    for key, val in history.items():
        avgd = np.mean(val[-avg_last_N_epochs:])
        print(f'\t{key} {avgd:g}')


def serial_sample_generator(model, action, batch_size, N_samples):
    layers, prior = model['layers'], model['prior']
    layers.eval()
    x, logq, logp = None, None, None
    for i in range(N_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
            logp = -action(x)
        yield x[batch_i], logq[batch_i], logp[batch_i]


def make_mcmc_ensemble(model, action, batch_size, N_samples):
    history = {
        'x' : [],
        'logq' : [],
        'logp' : [],
        'accepted' : []
    }

    # build Markov chain
    sample_gen = serial_sample_generator(model, action, batch_size, N_samples)
    for new_x, new_logq, new_logp in sample_gen:
        if len(history['logp']) == 0:
            # always accept first proposal, Markov chain must start somewhere
            accepted = True
        else:
            # Metropolis acceptance condition
            last_logp = history['logp'][-1]
            last_logq = history['logq'][-1]
            p_accept = torch.exp((new_logp - new_logq) - (last_logp - last_logq))
            p_accept = min(1, p_accept)
            draw = torch.rand(1) # ~ [0,1]
            if draw < p_accept:
                accepted = True
            else:
                accepted = False
                new_x = history['x'][-1]
                new_logp = last_logp
                new_logq = last_logq
        # Update Markov chain
        history['logp'].append(new_logp)
        history['logq'].append(new_logq)
        history['x'].append(new_x)
        history['accepted'].append(accepted)

    return history


def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    assert packaging.version.parse(torch.__version__) >= packaging.version.parse('1.5.0')
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
    padding_size = (kernel_size // 2)
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Conv2d(
            sizes[i], sizes[i+1], kernel_size, padding=padding_size,
            stride=1, padding_mode='circular'))
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())

    return torch.nn.Sequential(*net)


def set_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        torch.nn.init.normal_(m.weight, mean=1, std=2)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.fill_(-1)


def calc_dkl(logp, logq):
    return (logq - logp).mean()  # reverse KL, assuming samples from q


def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in coupling_layers:
        x, logJ = layer.forward(x)
        logq = logq - logJ
    return x, logq


def train_step(model, action, loss_fn, optimizer, metrics, batch_size):
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()

    x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = calc_dkl(logp, logq)
    loss.backward()

    optimizer.step()

    metrics['loss'].append(grab(loss))
    metrics['logp'].append(grab(logp))
    metrics['logq'].append(grab(logq))
    metrics['ess'].append(grab( compute_ess(logp, logq) ))


def compute_u1_plaq(links, mu, nu):
    """Compute U(1) plaqs in the (mu,nu) plane given `links` = arg(U)"""
    return (links[:,mu] + torch.roll(links[:,nu], -1, mu+1)
            - torch.roll(links[:,mu], -1, nu+1) - links[:,nu])


class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta
    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu+1,Nd):
                action_density = action_density + torch.cos(
                    compute_u1_plaq(cfgs, mu, nu))
        return -self.beta * torch.sum(action_density, dim=tuple(range(1,Nd+1)))


def gauge_transform(links, alpha):
    for mu in range(len(links.shape[2:])):
        links[:,mu] = alpha + links[:,mu] - torch.roll(alpha, -1, mu+1)
    return links


def random_gauge_transform(x):
    Nconf, VolShape = x.shape[0], x.shape[2:]
    return gauge_transform(x, 2*np.pi*torch.rand((Nconf,) + VolShape))


def topo_charge(x):
    P01 = torch_wrap(compute_u1_plaq(x, mu=0, nu=1))
    axes = tuple(range(1, len(P01.shape)))
    return torch.sum(P01, dim=axes) / (2*np.pi)


class MultivariateUniform(torch.nn.Module):
    """Uniformly draw samples from [a,b]"""
    def __init__(self, a, b):
        super().__init__()
        self.dist = torch.distributions.uniform.Uniform(a, b)
    def log_prob(self, x):
        axes = range(1, len(x.shape))
        return torch.sum(self.dist.log_prob(x), dim=tuple(axes))
    def sample_n(self, batch_size):
        return self.dist.sample((batch_size,))


class GaugeEquivCouplingLayer(torch.nn.Module):
    """U(1) gauge equiv coupling layer.

    Defined by `plaq_coupling` acting on plaquettes.
    """
    def __init__(self, *, lattice_shape, mask_mu, mask_off, plaq_coupling):
        super().__init__()
        link_mask_shape = (len(lattice_shape),) + lattice_shape
        self.active_mask = make_2d_link_active_stripes(link_mask_shape,
                                                       mask_mu, mask_off)
        self.plaq_coupling = plaq_coupling

    def forward(self, x):
        plaq = compute_u1_plaq(x, mu=0, nu=1)
        new_plaq, logJ = self.plaq_coupling(plaq)
        delta_plaq = new_plaq - plaq
        #  signs for U vs Udagger
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1)
        fx = (self.active_mask * torch_mod(delta_links + x)
              + (1-self.active_mask) * x)

        return fx, logJ

    def reverse(self, fx):
        new_plaq = compute_u1_plaq(fx, mu=0, nu=1)
        plaq, logJ = self.plaq_coupling.reverse(new_plaq)
        delta_plaq = plaq - new_plaq

        # signs for U vs Udagger
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1)
        x = (self.active_mask * torch_mod(delta_links + fx)
             + (1 - self.active_mask) * fx)

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
        mask[mu,:,0::4] = 1
    elif mu == 1:
        mask[mu,0::4] = 1
    nu = 1-mu
    mask = np.roll(mask, off, axis=nu+1)

    return torch.from_numpy(mask.astype(float_dtype)).to(torch_device)


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
    return torch.from_numpy(mask).to(torch_device)


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
    return torch.from_numpy(mask).to(torch_device)


def make_plaq_masks(mask_shape, mask_mu, mask_off):
    mask = {}
    mask['frozen'] = make_double_stripes(mask_shape, mask_mu, mask_off+1)
    mask['active'] = make_single_stripes(mask_shape, mask_mu, mask_off)
    mask['passive'] = 1 - mask['frozen'] - mask['active']

    return mask


def tan_transform(x, s):
    return torch_mod(2*torch.atan(torch.exp(s)*torch.tan(x/2)))


def tan_transform_logJ(x, s):
    return -torch.log(torch.exp(-s)*torch.cos(x/2)**2 + torch.exp(s)*torch.sin(x/2)**2)


def mixture_tan_transform(x, s):
    assert len(x.shape) == len(s.shape), \
        f'Dimension mismatch between x and s {x.shape} vs {s.shape}'
    return torch.mean(tan_transform(x, s), dim=1, keepdim=True)


def mixture_tan_transform_logJ(x, s):
    assert len(x.shape) == len(s.shape), \
        f'Dimension mismatch between x and s {x.shape} vs {s.shape}'
    return torch.logsumexp(tan_transform_logJ(x, s), dim=1) - np.log(s.shape[1])


def invert_transform_bisect(y, *, f, tol, max_iter, a=0, b=2*np.pi):
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
            if torch.all((mid_x == min_x) + (mid_x == max_x)):
                print('WARNING: Reached floating point precision before '
                      f'tolerance (iter {i}, err {err})')
                return mid_x
            min_x = greater_mask*mid_x + (1-greater_mask)*min_x
            min_val = greater_mask*mid_val + (1-greater_mask)*min_val
            max_x = (1-greater_mask)*mid_x + greater_mask*max_x
            max_val = (1-greater_mask)*mid_val + greater_mask*max_val
        print(f'WARNING: Did not converge to tol '
              f'{tol} in {max_iter} iters!  Error was {err}')

        return mid_x

def stack_cos_sin(x):
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


class NCPPlaqCouplingLayer(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_mu, mask_off,
                 inv_prec=1e-6, inv_max_iter=1000):
        super().__init__()
        assert len(mask_shape) == 2, (
            f'NCPPlaqCouplingLayer is implemented only in 2D, '
            f'mask shape {mask_shape} is invalid')
        self.mask = make_plaq_masks(mask_shape, mask_mu, mask_off)
        self.net = net
        self.inv_prec = inv_prec
        self.inv_max_iter = inv_max_iter

    def forward(self, x):
        x2 = self.mask['frozen'] * x
        net_out = self.net(stack_cos_sin(x2))
        assert net_out.shape[1] >= 2, (
            'CNN must output n_mix (s_i) + 1 (t) channels'
        )
        s, t = net_out[:,:-1], net_out[:,-1]

        x1 = self.mask['active'] * x
        x1 = x1.unsqueeze(1)
        local_logJ = self.mask['active'] * mixture_tan_transform_logJ(x1, s)
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)
        fx1 = self.mask['active'] * mixture_tan_transform(x1, s).squeeze(1)

        fx = (
            self.mask['active'] * torch_mod(fx1 + t) +
            self.mask['passive'] * x +
            self.mask['frozen'] * x
        )

        return fx, logJ

    def reverse(self, fx):
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
            self.mask['active'] * x1 +
            self.mask['passive'] * fx +
            self.mask['frozen'] * fx2
        )
        return x, logJ


def make_u1_equiv_layers(*, n_layers, n_mixture_comps,
                         lattice_shape, hidden_sizes, kernel_size):
    layers = []
    for i in range(n_layers):
        # periodically loop through all arrangements of maskings
        mu = i % 2
        off = (i//2) % 4
        in_channels = 2 # x - > (cos(x), sin(x))
        out_channels = n_mixture_comps + 1 # for mixture s and t, respectively
        net = make_conv_net(in_channels=in_channels, out_channels=out_channels,
                            hidden_sizes=hidden_sizes, kernel_size=kernel_size,
                            use_final_tanh=False)
        plaq_coupling = NCPPlaqCouplingLayer(
            net, mask_shape=lattice_shape, mask_mu=mu, mask_off=off
        )
        link_coupling = GaugeEquivCouplingLayer(
            lattice_shape=lattice_shape, mask_mu=mu, mask_off=off,
            plaq_coupling=plaq_coupling
        )
        layers.append(link_coupling)
    return torch.nn.ModuleList(layers)


# packaged from original ipynb by Xiao-Yong Jin
def flow_train(param):
    # Theory
    lattice_shape = param.lat
    link_shape = (2,*param.lat)
    beta = param.beta
    u1_action = U1GaugeAction(beta)

    # Model
    prior = MultivariateUniform(torch.zeros(link_shape), 2*np.pi*torch.ones(link_shape))
    n_layers = 16
    n_s_nets = 2
    hidden_sizes = [8,8]
    kernel_size = 3
    layers = make_u1_equiv_layers(lattice_shape=lattice_shape,
                                  n_layers=n_layers,
                                  n_mixture_comps=n_s_nets,
                                  hidden_sizes=hidden_sizes,
                                  kernel_size=kernel_size)
    set_weights(layers)
    model = {'layers': layers, 'prior': prior}

    # Training
    base_lr = .001
    optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)
    N_era = 10
    N_epoch = 100
    batch_size = 64
    print_freq = N_epoch # epochs
    plot_freq = 1 # epochs
    history = {
        'loss' : [],
        'logp' : [],
        'logq' : [],
        'ess' : []
    }
    for era in range(N_era):
        for epoch in range(N_epoch):
            train_step(model, u1_action, calc_dkl,
                       optimizer, history, batch_size)
            if epoch % print_freq == 0:
                print_metrics(history, print_freq, era, epoch)

    return model,u1_action


# packaged from original ipynb by Xiao-Yong Jin
def flow_eval(model, u1_action):
    ensemble_size = 8192
    u1_ens = make_mcmc_ensemble(model, u1_action, 64, ensemble_size)
    print("Accept rate:", np.mean(u1_ens['accepted']))
    Q = grab(topo_charge(torch.stack(u1_ens['x'], axis=0)))
    X_mean, X_err = bootstrap(Q**2, Nboot=100, binsize=16)
    print(f'Topological susceptibility = {X_mean:.2f} +/- {X_err:.2f}')
    print(f'... vs HMC estimate = 1.23 +/- 0.02')

# ------ END CODE FROM https://arxiv.org/abs/2101.08176

if __name__ == '__main__':
    param = Param(
        beta = 2.0,
        lat = (8, 8),
        tau = 2, # 0.3
        nstep = 8, # 3
        ntraj = 256, # 2**16 # 2**10 # 2**15
        nprint = 8,
        seed = 1331)

    torch.manual_seed(param.seed)

    torch.set_num_threads(param.nth)
    torch.set_num_interop_threads(param.nth_interop)
    os.environ["OMP_NUM_THREADS"] = str(param.nth)
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    torch.set_default_tensor_type(torch.DoubleTensor)

    def run(param):
        with open(param.uniquestr(), "w") as O:
            params = param.summary()
            O.write(params)
            put(params)
            field = param.initializer() # mu, x, t
            plaq, topo = (action(param, field) / (-param.beta*param.volume), topocharge(field))
            status = f"Initial configuration:  plaq: {plaq}  topo: {topo}\n"
            O.write(status)
            put(status)
            ts = []
            for n in range(param.nrun):
                t = -timer()
                for i in range(param.ntraj):
                    dH, exp_mdH, acc, field = hmc(param, field)
                    plaq = action(param, field) / (-param.beta*param.volume)
                    topo = topocharge(field)
                    ifacc = "ACCEPT" if acc else "REJECT"
                    status = f"Traj: {n*param.ntraj+i+1:4}  {ifacc}:  dH: {dH:< 12.8}  exp(-dH): {exp_mdH:< 12.8}  plaq: {plaq:< 12.8}  topo: {topo:< 3.3}\n"
                    O.write(status)
                    if (i+1) % (param.ntraj//param.nprint) == 0:
                        put(status)
                t += timer()
                ts.append(t)
            print("Run times: ", ts)
            print("Per trajectory: ", [t/param.ntraj for t in ts])
        return field
    field = run(param)
    field = torch.reshape(field,(1,)+field.shape)

    flow_model,flow_act = flow_train(param)
    flow_eval(flow_model,flow_act)

    flow = flow_model['layers']
    flow.eval()
    print(f'plaq(field) {action(param, field[0]) / (-param.beta*param.volume)}')
    field.requires_grad_(True)
    x = field
    logJ = 0.0
    for layer in reversed(flow):
        x, lJ = layer.reverse(x)
        logJ += lJ
    s = action(param, x[0]) - logJ
    s.backward()
    f = field.grad
    field.requires_grad_(False)
    print(
        f'plaq(x): {action(param, x[0]) / (-param.beta*param.volume)} '
        f'logJ: {logJ} '
        f'force.norm: {torch.linalg.norm(f)}'
    )
    y = x
    logJy = 0.0
    for layer in flow:
        y, lJ = layer.forward(y)
        logJy += lJ
    print(
        f'plaq(y): {action(param, y[0]) / (-param.beta*param.volume)} '
        f'logJy: {logJy}'
    )
