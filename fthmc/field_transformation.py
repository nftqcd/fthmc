from __future__ import absolute_import, division, print_function, annotations
import torch
import torch.nn as nn

from fthmc.utils import qed_helpers as qed
from fthmc.config import TrainConfig, Param


class FieldTransformation(nn.Module):
    def __init__(
            self,
            param: Param,
            flow: nn.ModuleList,
            config: TrainConfig = None
    ):
        super().__init__()
        self.param = param
        self.flow = flow
        self.config = config
        action_fn = qed.BatchAction(param.beta)
        self._action_fn = lambda x: action_fn(x)[0]
        self._charges_fn = lambda x: qed.batch_charges(x)[0]

    def action(self, x: torch.Tensor):
        #y = x

        #logdet = 0.
        #for layer in self.flow:
        #    y, logdet_ = layer(y)
        #    logdet = logdet + logdet_
        #
        #return self._action_fn(y) - logdet
        z, logdet = self.flow_forward(x)

        return self._action_fn(z) - logdet

    def flow_forward(self, x: torch.Tensor):
        logdet = 0.
        for layer in self.flow:
            x, logdet_ = layer(x)
            logdet = logdet + logdet_

        return x, logdet

    def flow_backward(self, x: torch.Tensor):
        logdet = 0.
        for layer in self.flow[::-1]:
            x, logdet_ = layer.reverse(x)
            logdet = logdet - logdet_

        return x, logdet

    def force(self, x: torch.Tensor, create_graph: bool = False):
        x.requires_grad_(True)
        s = self.action(x)
        dsdx, = torch.autograd.grad(s.sum(), x, create_graph=create_graph)
        x.requires_grad_(False)

        return dsdx

    def calc_metrics(self, x: torch.Tensor):
        logp = (-1.) * self._action_fn(x)
        plaq = logp / (self.param.beta * self.param.volume)
        q = qed.batch_charges(x)
        return q, plaq


    def leapfrog(self, x: torch.Tensor, v: torch.Tensor):
        dt = self.param.dt

        x = x + 0.5 * dt * v
        #force = self.force(x)
        v = v + (-dt) * self.force(x)

        #metrics = {'dt': [], 'acc': [], 'dqsq': [], 'plaq': []}
        #q, plaq = self.calc_metrics(x)

        #metrics['dt'].append(0)
        #metrics['acc'].append(1)
        #metrics['dqsq'].append(0)
        #metrics['plaq'].append(plaq)
        #metrics['q'].append(q)

        for _ in range(self.param.nstep - 1):
            x = x + dt * v
            #q, plaq = self.calc_metrics(x)
            v = v + (-dt) * self.force(x)

        x = x + 0.5 * dt * v
        return x, v

    @staticmethod
    def wrap(x: torch.Tensor):
        x_ = (x - PI) / TWO_PI

        return TWO_PI * (x_ - x_.floor() - 0.5)

    def build_trajectory(self, x: torch.Tensor = None):
        if x is None:
            x = self.param.initializer()


        x0 = x.clone()
        v0 = torch.randn_like(x)
        #v0_norm = (v0 * v0).sum()

        h0 = self.action(x) + 0.5 * (v0 * v0).sum()
        #z = self.action(x) + 0.5 * v0_norm

        x, v1 = self.leapfrog(x, v0)
        x1 = self.wrap(x)
        h1 = self.action(x1) + 0.5 * (v1 * v1).sum()
        #v1_norm = (v * v).sum()
        #plaq1 = self.action(xr) + 0.5 * v1_norm
        dh = h1 - h0
        exp_mdh = torch.exp(-dh)
        #prob = torch.minimum(torch.ones_like(exp_mdh), exp_mdh)
        acc = (torch.rand_like(exp_mdh) < exp_mdh).float()
        x_ = acc * x1 + (1 - acc) * x0


        #exp_mdh = torch.exp(-dh)
        #prob = torch.rand(x.shape[0])
        #acc = prob < exp_mdH
        #acc = torch.minimum(d)


        #prob = torch.rand([], dtype=torch.float64)
        #dH = plaq1 - plaq0
        #exp_mdH = torch.exp(-dH)
        #acc = prob < exp_mdH
        #x_ = xr if acc else x
        #for layer Iin self.flow:
        #    x_, logdet = layer(x_)

        return x_, acc, dH, exp_mdh


    def forward1(self, x: torch.Tensor):
        v0 = torch.randn_like(x)

        v0_norm = (v0 * v0).sum()
        logdet = 0.
        for layer in self.flow:
            x, logdet_ = layer(x)
            logdet = logdet + logdet_

        action = self._action_fn(x) - logdet

        plaq0 = self.action(x) + 0.5 * v0_norm

        x, v1 = self.leapfrog(x, v)
        xr = self.wrap(x)

        v1_norm = (v1 * v1).sum()
        plaq1 = self.action(x) + 0.5 * v1_norm

        prob = torch.rand(x.shape[0])
        dH = plaq1 - plaq0
        exp_mdH = torch.exp(-dH)
        acc = (prob < exp_mdH).float()
        x_ = xr if acc else x

        prob = torch.exp(torch.minimum(dH, torch.zeros_like(dH)))
        acc = torch.rand(prob.shape) < prob
        pass


    def run(self, x: torch.Tensor = None, nprint: int = 1, nplot: int = 1):
        if x is None:
            x = self.param.initializer()

        runs_history = {}
        fields = []
        beta = self.param.beta
        volume = self.param.volume

        #fig, ax = plt.subplots(constrained_layout=True)
        #fig1, ax1 = plt.subplots(constrained_layout=True)
        #fig2, ax2 = plt.subplots(constrained_layout=True)
        plots_acc = init_live_plot(figsize=(5, 1.),
                                   param=self.param,
                                   ylabel='acc',
                                   xlabel='trajectory',
                                   config=self.config)
        plots_q = init_live_plot(figsize=(5, 1.),
                                 #param=self.param,
                                 ylabel='q',
                                 xlabel='trajectory')
                                 #config=self.config)
        plots_plaq = init_live_plot(figsize=(5, 1.),
                                    ylabel='plaq',
                                    xlabel='trajectory')
                                    #param=self.param,
                                    #config=self.config)
        plots = {'plaq': plots_plaq, 'q': plots_q, 'acc': plots_acc}

        for n in range(self.param.nrun):
            t0 = time.time()
            xarr = []
            history = {}
            for i in range(param.ntraj):
                t_ = time.time()
                q0 = self._charges_fn(x)

        for n in range(self.param.nrun):
            t0 = time.time()
            xarr = []
            history = {}
            for i in range(param.ntraj):
                t_ = time.time()
                q0 = self._charges_fn(x)
                #q0 = qed.batch_charges(x[None, :])

                x, acc, dH, exp_mdH = self.build_trajectory(x)
                #x = self.
                q1 = self._charges_fn(x)
                #q1 = qed.batch_charges(x[None, :])

                dqsq = (q1 - q0) ** 2
                #logp = (-1.) * self._action_fn(x)
                logp = (-1.) * self.action(x)
                plaq = logp / (beta * volume)
                plaq_no_grad = plaq.detach()

                metrics = {
                    'traj': i,
                    'dt': time.time() - t_,
                    'acc': True if acc else False,
                    'dH': dH,
                    'exp_mdH': exp_mdH,
                    'dqsq': dqsq,
                    'q': q1,
                    'plaq': plaq_no_grad,
                }

                for key, val in metrics.items():
                    try:
                        history[key].append(val)
                    except KeyError:
                        history[key] = [val]

                if i % nplot == 0:
                    data = {
                        'q': history['q'],
                        'acc': history['acc'],
                        'plaq': history['plaq']
                    }
                    update_plots(plots, data)


                if i % nprint == 0:
                    logger.print_metrics(metrics, skip=['q'])#, pre=['(now)'])
                #logger.print_metrics(metrics, skip=['q'], pre=['(avg)'],
                #                     window=min(i, 20))

            runs_history[n] = history

        return runs_history
