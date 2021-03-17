"""
qed_lattice.py

Contains implementation of `qedLattice` object.
"""
import sys

from math import pi as PI
from timeit import default_timer as timer
from dataclasses import dataclass
from numpy.random import f

import torch
import numpy as np

from utils.param import Param
from utils.plot_helpers import init_live_plot, update_plots
from utils.distributions import MultivariateUniform
from torch.distributions.uniform import Uniform
from torch.distributions import distribution
from torch.optim import Adam
from utils.field_transformation import (make_u1_equiv_layers,
                                        make_mcmc_ensemble, bootstrap)

TWO_PI = 2 * PI

def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


def print_metrics(history, avg_last_N_epochs, era, epoch):
    print(f'== Era {era} | Epoch {epoch} metrics ==')
    for key, val in history.items():
        avgd = np.mean(val[-avg_last_N_epochs:])
        print(f'\t{key} {avgd:g}')


def put(s, end=None):
    if end is not None:
        s += str(end)

    sys.stdout.write(s)


def put_and_write(s, out, end=None):
    if end is not None:
        s += str(end)

    out.write(s)
    put(s)

    return s


def calc_dkl(logp, logq):
    return (logq - logp).mean()  # reverse KL, assuming samples from q


def compute_ess(logp: torch.Tensor, logq: torch.Tensor):
    logw = logp - logq
    log_ess = (2 * torch.logsumexp(logw, dim=0)
               - torch.logsumexp(2*logw, dim=0))
    ess_per_cfg = torch.exp(log_ess) / len(logw)

    return ess_per_cfg


def apply_flow_to_prior(prior, layers, *, batch_size):
    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)
    for layer in layers:
        x, logJ = layer.forward(x)
        logq = logq - logJ

    return x, logq


def serial_sample_generator(model, param, batch_size, num_samples):
    layers = model['layers']
    prior = model['prior']
    layers.eval()
    x, logq, logp = None, None, None
    for i in range(num_samples):
        batch_i = i % batch_size
        if batch_i == 0:
            # we're out of samples to propose, generate a new batch
            x, logq =


@dataclass
class FlowConfig:
    n_layers: int
    n_s_nets: int
    hidden_sizes: list
    kernel_size: int
    batch_size: int


from torch.autograd import grad
class qedLattice:
    def __init__(self, param: Param):
        self.param = param
        self._beta = param.beta

    @staticmethod
    def regularize(field):
        field_ = (field - PI) / TWO_PI
        return TWO_PI * (field_ - torch.floor(field_) - 0.5)

    @staticmethod
    def plaq_phase(field):
        return (field[0, :]
                - field[1, :]
                - torch.roll(field[0, :], shifts=-1, dims=1)
                + torch.roll(field[1, :], shifts=-1, dims=0))

    def action(self, field):
        phase = self.plaq_phase(field)
        return (-self._beta) * torch.sum(torch.cos(phase)) / TWO_PI

    def topo_charge(self, field):
        reg_phase = self.regularize(self.plaq_phase(field))
        return torch.floor(0.1 + torch.sum(reg_phase) / TWO_PI)

    def force(self, field: torch.Tensor):
        field.requires_grad_(True)
        s = self.action(field)
        field.grad = None
        s.backward()
        df = field.grad
        field.requires_grad_(False)

        return df

    def ft_action1(self, field, flow):
        logdet = 0.
        for layer in flow:
            field, logdet_ = layer.forward(field)
            logdet += logdet_

        return self.action(field) - logdet

    def ft_force(self, field, flow, create_graph=False):
        # field is the field from the transformed distribution
        # close to prior
        f = torch.tensor(field, requires_grad=True)
        s = self.ft_action(f, flow)
        ss = torch.sum(s)
        df, = torch.autograd.grad(ss, f, create_graph=create_graph)
        #  ss.backward(gradient=f)
        #  df = field.grad
        #  df, = torch.autograd.grad(ss, f, create_graph=create_graph)
        #  f.requires_grad_(False)

        #  df, = torch.autograd.grad(ss, field, create_graph=create_graph)
        #  field.requires_grad_(False)

        return df

    @staticmethod
    def ft_flow_inv1(field, flow):
        logdet = 0.0
        for layer in reversed(flow):
            field, logdet_ = layer.reverse(field)
            logdet += logdet_

        return field.detach(), logdet

    @staticmethod
    def ft_flow1(field, flow):
        logdet = 0.0
        for layer in flow:
            field, logdet_ = layer.forward(field)
            logdet += logdet_

        return field.detach(), logdet

    @staticmethod
    def ft_flow(field, flow):
        for layer in flow:
            field, _ = layer.forward(field)

        return field.detach

    @staticmethod
    def ft_flow_inv(field, flow):
        for layer in reversed(flow):
            field, _ = layer.reverse(field)

        return field.detach()

    def ft_action(self, field, flow):
        y = field
        logJ = 0.0
        for layer in flow:
            y, logJ_ = layer.forward(y)
            logJ += logJ_

        s = self.action(y) - logJ
        return s

    def ft_force2(self, field, flow, create_graph=True):
        f = field
        f.requires_grad_(True)
        s = self.ft_action(f, flow)
        ss = torch.sum(s)
        ff, = torch.autograd.grad(ss, f, create_graph=create_graph)
        f.requires_grad_(False)

        return ff

    def leapfrog(self, x, p, verbose=True):
        dt = self.param.dt
        x_ = x + 0.5 * dt * p
        f = self.force(x_)
        p_ = p + (-dt) * f
        if verbose:
            denom = (-self._beta * self.param.volume)
            print(f'plaq(x): {self.action(x) / denom:<.4g}, '
                  f'force.norm: {torch.linalg.norm(f):>.4g}')

        for _ in range(self.param.nstep - 1):
            x_ = x_ + dt * p_
            p_ = p_ + (-dt) * self.force(x_)

        x_ = x_ + 0.5 * dt * p_

        return (x_, p_)

    def serial_sample_generator(self, model, batch_size, num_samples):
        layers, prior = model['layers'], model['prior']
        layers.eval()
        x, logq, logp = None, None, None
        for i in range(num_samples):
            batch_i = i % batch_size
            if batch_i == 0:
                # we're out of samples to propose, generate a new batch
                x, logq = apply_flow_to_prior(prior, layers,
                                              batch_size=batch_size)
                logp = -self.action(x)

            yield x[batch_i], logq[batch_i], logp[batch_i]

    def make_mcmc_ensemble(self, model, batch_size, num_samples):
        history = {
            'x': [],
            'logq': [],
            'logp': [],
            'accepted': [],
        }

        # build Markov chain
        sample_gen = self.serial_sample_generator(model, batch_size=batch_size,
                                                  num_samples=num_samples)
        for new_x, new_logq, new_logp in sample_gen:
            if len(history['logp']) == 0:
                # always accept first prooposal, Markov chain must start
                # somewhere
                accepted = True
            else:
                # Metropolis acceptance criteria
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
            # Update Markov chain
            history['logp'].append(new_logp)
            history['logq'].append(new_logq)
            history['x'].append(new_x)
            history['accepted'].append(accepted)

        return history

    def hmc(self, x, verbose=True):
        p = torch.rand_like(x)
        act0 = self.action(x) + 0.5 * torch.sum(p * p)
        x_, p_ = self.leapfrog(x, p, verbose)
        xr = self.regularize(x_)
        act = self.action(xr) + 0.5 * torch.sum(p_ * p_)
        prob = torch.rand([], dtype=torch.float64)
        dH = act - act0
        exp_dH = torch.exp(-dH)
        #  exp_mdH = torch.exp(-dH)
        acc = prob < exp_dH
        newx = xr if acc else x

        return (dH, exp_dH, acc, newx)

    def run(self, field=None, verbose=True):
        if field is None:
            field = self.param.initializer()

        beta_factor = (-self._beta * self.param.volume)
        with open(self.param.uniquestr(), 'w') as fout:
            summary = self.param.summary()
            fout.write(summary+'\n')
            put(summary, end='\n')
            #  put_and_write(summary, fout, end='\n')
            plaq = self.action(field) / beta_factor
            topo = self.topo_charge(field)
            status = f'Initial configuration: plaq: {plaq}, topo: {topo}\n'
            put(status, end='\n')
            fout.write(status + '\n')
            #  put_and_write(status, fout, end='\n')

            ts = []
            for n in range(self.param.nrun):
                t = - timer()
                for i in range(self.param.ntraj):
                    dh, expdh, acc, field = self.hmc(field, verbose)
                    plaq = self.action(field) /beta_factor
                    topo = self.topo_charge(field)

                    ifacc = 'ACCEPT' if acc else 'REJECT'
                    status = {
                        'traj': f'{n * self.param.ntraj+i+1:4}',
                        'mh': f'{ifacc}',
                        'dH': f'{dh:< 12.8}',
                        'exp(-dH)': f'{expdh:< 12.8}',
                        'plaq': f'{plaq:< 12.8}',
                        'topo': f'{topo:< 3.3}\n',
                    }
                    s = ', '.join(
                        '='.join((k, v)) for k, v in status.items()
                    )
                    fout.write(s + '\n')
                    if (i + 1) % (self.param.ntraj // self.param.nprint) == 0:
                        put(s, end='\n')

                t += timer()
                ts.append(t)

            rstr = f'Run times: {ts}\n'
            rstr1 = (
                f'Per trajectory: {[t / self.param.ntraj for t in ts]}\n'
            )
            put(rstr)
            put(rstr1)
            fout.write(rstr)
            fout.write(rstr1)
            #  put_and_write(f'Run times: {ts}', '\n')
            #  print(f'Per trajectory: {[t / self.param.ntraj for t in ts]}')
        return field


    def train_step(
            self,
            model,
            optimizer,
            metrics,
            batch_size,
            with_force=False,
            pre_model=None,  # Why is this needed?
            create_graph=True,
    ):
        prior = model['prior']
        layers = model['layers']

        optimizer.zero_grad()

        if pre_model != None:
            pre_prior = pre_model['prior']
            pre_layers = pre_model['layers']
            pre_xi = pre_prior.sample_n(batch_size)
            x, pre_logJ = self.ft_flow(pre_xi, pre_layers)
            xi, pre_logJ_inv = self.ft_flow_inv(x, pre_layers)

        x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)

        logp = -self.action(x)

        dkl = calc_dkl(logp, logq)

        loss = torch.tensor(0.0)
        force_size = torch.tensor(0.0)
        if with_force:
            assert pre_model != None
            force = self.ft_force(x, layers, create_graph=create_graph)
            force_size = torch.sum(torch.square(force))
            loss = force_size
        else:
            loss = dkl

        loss.backward()
        optimizer.step()
        ess = compute_ess(logp, logq)

        force_norm = torch.linalg.norm(
            self.ft_force(x, layers, create_graph=create_graph)
        )
        metrics_ = {
            'loss': grab(loss),
            'force': grab(force_size),
            'dkl': grab(dkl),
            'ess': grab(ess),
            'force.norm': force_norm,
        }
        s = ', '.join('='.join((str(k), str(v))) for k, v in metrics_.items())
        print(f'{s}\n')
        for key, val in metrics_.items():
            if key in metrics:
                metrics[key].append(val)

    def flow_train(
            self,
            with_force: bool = False,
            pre_model: dict = None,
            flow_config: FlowConfig = None,
            create_graph: bool = True,
            base_lr: float = 0.001,
            n_era: int = 10,
            n_epoch: int = 100,
            plot_freq: int = 0,
    ):
        lattice_shape = self.param.lat
        link_shape = (2, *self.param.lat)
        if flow_config is None:
            flow_config = FlowConfig(n_layers=8,
                                     n_s_nets=4,
                                     hidden_sizes=[8, 8],
                                     batch_size=64,
                                     kernel_size=3)

        prior = MultivariateUniform(torch.zeros(link_shape),
                                    2 * np.pi * torch.ones(link_shape))
        layers = make_u1_equiv_layers(lattice_shape=lattice_shape,
                                      n_layers=flow_config.n_layers,
                                      n_mixture_comps=flow_config.n_s_nets,
                                      hidden_sizes=flow_config.hidden_sizes,
                                      kernel_size=flow_config.kernel_size)

        model = {'layers': layers, 'prior': prior}
        optimizer = Adam(model['layers'].parameters(), lr=base_lr)
        optimizer_wf = Adam(model['layers'].parameters(), lr=base_lr / 100.0)
        print_freq = n_epoch
        history = {
            'loss': [], 'force': [],
            'dkl': [], 'logp': [], 'logq': [], 'ess': [],
        }
        if plot_freq > 0:
            plot_dict = init_live_plot(n_era, n_epoch)

        for era in range(n_era):
            for epoch in range(n_epoch):
                self.train_step(model=model,
                                optimizer=optimizer,
                                metrics=history,
                                batch_size=flow_config.batch_size,
                                with_force=False,
                                pre_model=None,
                                create_graph=create_graph)
                if epoch % plot_freq == 0:
                    update_plots(history, window=15, **plot_dict)

                if with_force:
                    self.train_step(model=model,
                                    optimizer=optimizer_wf,
                                    metrics=history,
                                    batch_size=flow_config.batch_size,
                                    with_force=with_force,
                                    pre_model=pre_model)

                if epoch % print_freq == 0:
                    print_metrics(history, print_freq, era, epoch)

        return model

    def flow_eval(self, model, ensemble_size, batch_size):
        ensemble = make_mcmc_ensemble(model, self.action, batch_size,
                                      ensemble_size)
        print(f'Accept rate: {np.mean(ensemble["accepted"]): >12.4g}')
        Q = grab(self.topo_charge(torch.stack(ensemble['x'], axis=0)))
        X_mean, X_err = bootstrap(Q ** 2, nboot=100, binsize=16)
        print(f'Topological susceptibility: {X_mean:.2f} +/- {X_err:.2f}')
        print(f'... vs HMC estimate = 1.23 +/- 0.02')
