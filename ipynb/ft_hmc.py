#!/usr/bin/env python3

import torch
import math
import sys
import os
import numpy as np
from timeit import default_timer as timer
from functools import reduce
from field_transformation import *

# statistics

def flattern(l):
    return [x for y in l for x in y]

def average(l):
    return sum(l) / len(l)

def sigma(l):
    avg = average(l)
    sq_avg = average([ np.square(v - avg) for v in l ])
    return sq_avg / np.sqrt(len(l) - 1)

def sub_avg(l):
    avg = average(l)
    return np.array([x - avg for x in l])

n_block = 16 # ADJUST ME

def block_list(l):
    n_block_local = n_block
    size_block = len(l) // n_block_local
    if size_block < 1:
        size_block = 1
        n_block_local = len(l)
    if n_block_local == 0:
        return []
    start = len(l) - n_block_local * size_block
    return [ l[ start + i * size_block : start + (i+1) * size_block] for i in range(n_block_local) ]

def change_sqr(l, lp):
    size = min(len(l), len(lp))
    if size == 0:
        return []
    vs = [ np.square(l[i] - lp[i]) for i in range(size) ]
    vs = list(map(average, block_list(vs)))
    avg = average(vs)
    sig = sigma(vs)
    return [ avg, sig ]

def change_sqr_vs_dt(l, dt_range = 10):
    return [ [ i ] + change_sqr(l, l[i:]) for i in range(1, dt_range + 1)]

# From Xiao-Yong

class Param:
    def __init__(self, beta = 6.0, lat = (64, 64), tau = 2.0, nstep = 50, ntraj = 256, nrun = 4, nprint = 256, seed = 11*13, randinit = False, nth = int(os.environ.get('OMP_NUM_THREADS', '2')), nth_interop = 2):
        self.beta = beta
        self.lat = lat
        self.nd = len(lat)
        self.volume = reduce(lambda x,y:x*y, lat)
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
            return torch.empty((self.nd,) + self.lat).uniform_(-math.pi, math.pi)
        else:
            return torch.zeros((self.nd,) + self.lat)
    def summary(self):
        return f"""latsize = {self.lat}
volume = {self.volume}
beta = {self.beta}
trajs = {self.ntraj}
tau = {self.tau}
steps = {self.nstep}
seed = {self.seed}
nth = {self.nth}
nth_interop = {self.nth_interop}
"""
    def uniquestr(self, tag = ""):
        lat = ".".join(str(x) for x in self.lat)
        return f"out_b{self.beta}_l{lat}_n{self.ntraj}_t{self.tau}_s{self.nstep}{tag}.out"

def action(param, f):
    return (-param.beta)*torch.sum(torch.cos(plaqphase(f)))

def force(param, f):
    f.requires_grad_(True)
    s = action(param, f)
    f.grad = None
    s.backward()
    ff = f.grad
    f.requires_grad_(False)
    return ff

plaqphase = lambda f: f[0,:] - f[1,:] - torch.roll(f[0,:], shifts=-1, dims=1) + torch.roll(f[1,:], shifts=-1, dims=0)

topocharge = lambda f: torch.floor(0.1 + torch.sum(regularize(plaqphase(f))) / (2*math.pi))

def regularize(f):
    p2 = 2*math.pi
    f_ = (f - math.pi) / p2
    return p2*(f_ - torch.floor(f_) - 0.5)

hmc_info_list = []

def leapfrog(param, x, p):
    mom_norm = torch.sum(p*p)
    info_list = []
    dt = param.dt
    x_ = x + 0.5*dt*p
    f = force(param, x_)
    p_ = p + (-dt)*f
    info = np.array((float(torch.linalg.norm(f)),
                     float(action(param, x_)),
                     float(torch.sum(p*p_)/np.sqrt(mom_norm*torch.sum(p_*p_)))))
    info_list.append(info)
    print(f'plaq(x) {action(param, x) / (-param.beta*param.volume)}  force.norm {torch.linalg.norm(f)}')
    for i in range(param.nstep-1):
        x_ = x_ + dt*p_
        f = force(param, x_)
        info = np.array((float(torch.linalg.norm(f)),
                        float(action(param, x_)),
                        float(torch.sum(p*p_)/np.sqrt(mom_norm*torch.sum(p_*p_)))))
        info_list.append(info)
        p_ = p_ + (-dt)*f
    x_ = x_ + 0.5*dt*p_
    print(np.sqrt(average([l[0]**2 for l in info_list])),
          (info_list[0][1], info_list[-1][1]),
          info_list[-1][2])
    hmc_info_list.append(info_list)
    return (x_, p_)

def hmc(param, x):
    p = torch.randn_like(x)
    act0 = action(param, x) + 0.5*torch.sum(p*p)
    x_, p_ = leapfrog(param, x, p)
    xr = regularize(x_)
    act = action(param, xr) + 0.5*torch.sum(p_*p_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act-act0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    newx = xr if acc else x
    return (dH, exp_mdH, acc, newx)

put = lambda s: sys.stdout.write(s)

def show_hmc_stats():
    fl_list = flattern(hmc_info_list)
    l_len = len(fl_list)
    fl_list = fl_list[l_len // 2:]
    action_list = np.array([l[1] for l in fl_list])
    action_list = sub_avg(action_list)
    print("action sigma", np.sqrt(average(action_list**2)))
    force_list = np.array([l[0] for l in fl_list])
    print("avg force", np.sqrt(average(force_list**2)))

def save_topo_change_sqr(fn):
    size_hist = len(topo_history)
    drop_len = size_hist // 3 # ADJUST ME
    with open(fn, 'w') as f:
        dt = change_sqr_vs_dt(topo_history[drop_len:])
        for l in dt:
            print(l)
            f.write(f"{l[0]} {l[1]} {l[2]}\n")
    sys.stdout.flush()

topo_history = []

def run(param, field = None):
    if field is None:
        field = param.initializer()
    hmc_info_list.clear()
    topo_history.clear()
    fn = param.uniquestr()
    if os.path.exists(fn):
        return field
    with open(fn, "w") as O:
        params = param.summary()
        O.write(params)
        put(params)
        plaq, topo = (action(param, field) / (-param.beta*param.volume), topocharge(field))
        status = f"Initial configuration:  plaq: {plaq}  topo: {topo} {field.shape}\n"
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
                status = f"Traj: {n*param.ntraj+i+1:4}  {ifacc}  dH: {dH:< 12.8}  exp(-dH): {exp_mdH:< 12.8}  plaq: {plaq:< 12.8}  topo: {topo:< 3.3}\n"
                O.write(status)
                put(status)
                topo_history.append(topo.item())
                sys.stdout.flush()
            t += timer()
            ts.append(t)
        print("Run times: ", ts)
        print("Per trajectory: ", [t/param.ntraj for t in ts])
    show_hmc_stats()
    fn = param.uniquestr("_topo")
    save_topo_change_sqr(fn)
    return field

# End from Xiao-Yong

def ft_flow(flow, f):
    for layer in flow:
        f, lJ = layer.forward(f)
    return f.detach()

def ft_flow_inv(flow, f):
    for layer in reversed(flow):
        f, lJ = layer.reverse(f)
    return f.detach()

def ft_action(param, flow, f):
    y = f
    logJy = 0.0
    for layer in flow:
        y, lJ = layer.forward(y)
        logJy += lJ
    action = U1GaugeAction(param.beta)
    s = action(y) - logJy
    return s

def ft_force(param, flow, field, create_graph = False):
    # f is the field follows the transformed distribution (close to prior distribution)
    f = field
    f.requires_grad_(True)
    s = ft_action(param, flow, f)
    ss = torch.sum(s)
    # f.grad = None
    ff, = torch.autograd.grad(ss, f, create_graph = create_graph)
    f.requires_grad_(False)
    return ff

# train model

def train_step(model, action, optimizer, metrics, batch_size, param, with_force = False, pre_model = None):
    layers, prior = model['layers'], model['prior']
    optimizer.zero_grad()
    #
    xi = None
    if pre_model != None:
        pre_layers, pre_prior = pre_model['layers'], pre_model['prior']
        pre_xi = pre_prior.sample_n(batch_size)
        x = ft_flow(pre_layers, pre_xi)
        xi = ft_flow_inv(layers, x)
    #
    xi, x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size, xi=xi)
    logp = -action(x)
    #
    force_size = torch.tensor(0.0)
    dkl = calc_dkl(logp, logq)
    loss = torch.tensor(0.0)
    if with_force:
        assert pre_model != None
        force = ft_force(param, layers, xi, True)
        force_size = torch.sum(torch.square(force))
        loss = force_size
    else:
        loss = dkl
    #
    loss.backward()
    #
    # minimization target
    # loss mini
    # -> (logq - logp) mini
    # -> (action - logJ) mini
    #
    optimizer.step()
    ess = compute_ess(logp, logq)
    #
    print(grab(loss),
          grab(force_size),
          grab(dkl),
          grab(ess),
          torch.linalg.norm(ft_force(param, layers, xi)))
    #
    metrics['loss'].append(grab(loss))
    metrics['force'].append(grab(force_size))
    metrics['dkl'].append(grab(dkl))
    metrics['logp'].append(grab(logp))
    metrics['logq'].append(grab(logq))
    metrics['ess'].append(grab(ess))

def flow_train(param, with_force = False, pre_model = None):  # packaged from original ipynb by Xiao-Yong Jin
    # Theory
    lattice_shape = param.lat
    link_shape = (2,*param.lat)
    beta = param.beta
    u1_action = U1GaugeAction(beta)
    # Model
    prior = MultivariateUniform(torch.zeros(link_shape), 2*np.pi*torch.ones(link_shape))
    #
    n_layers = 24
    n_s_nets = 2
    hidden_sizes = [8,8]
    kernel_size = 3
    layers = make_u1_equiv_layers(lattice_shape=lattice_shape, n_layers=n_layers, n_mixture_comps=n_s_nets,
                                  hidden_sizes=hidden_sizes, kernel_size=kernel_size)
    set_weights(layers)
    model = {'layers': layers, 'prior': prior}
    # Training
    base_lr = .0001
    optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr)
    optimizer_wf = torch.optim.Adam(model['layers'].parameters(), lr=base_lr / 100.0)
    #
    # ADJUST ME
    N_era = 10
    N_epoch = 30
    #
    batch_size = 64
    print_freq = N_epoch # epochs
    plot_freq = 1 # epochs
    history = {
        'loss' : [],
        'force' : [],
        'dkl' : [],
        'logp' : [],
        'logq' : [],
        'ess' : []
    }
    for era in range(N_era):
        for epoch in range(N_epoch):
            train_step(model, u1_action, optimizer, history, batch_size, param)
            if with_force:
                train_step(model, u1_action, optimizer_wf, history, batch_size, param,
                        with_force = with_force, pre_model = pre_model)
            if epoch % print_freq == 0:
                print_metrics(history, print_freq, era, epoch)
    return model,u1_action

def flow_eval(model, u1_action):  # packaged from original ipynb by Xiao-Yong Jin
    ensemble_size = 1024
    u1_ens = make_mcmc_ensemble(model, u1_action, 64, ensemble_size)
    print("Accept rate:", np.mean(u1_ens['accepted']))
    Q = grab(topo_charge(torch.stack(u1_ens['x'], axis=0)))
    X_mean, X_err = bootstrap(Q**2, Nboot=100, binsize=16)
    print(f'Topological susceptibility = {X_mean:.2f} +/- {X_err:.2f}')

def train_load_save(param):
    lat = "x".join(str(x) for x in param.lat)
    fn = f"flow_b{param.beta}_l{lat}.dat"
    if os.path.exists(fn):
        return torch.load(fn)
    pre_flow_model, flow_act = flow_train(param)
    flow_eval(pre_flow_model,flow_act)
    pre_flow = pre_flow_model['layers']
    train_force = False
    flow_model = None
    if train_force:
        flow_model, flow_act = flow_train(param, with_force=True, pre_model=pre_flow_model)
    else:
        flow_model = pre_flow_model
    flow_eval(flow_model,flow_act)
    flow = flow_model['layers']
    torch.save(flow, fn)
    return flow

# test

def test_force(x = None):
    model = flow_model
    layers, prior = model['layers'], model['prior']
    if x == None:
        pre_model = pre_flow_model
        pre_layers, pre_prior = pre_model['layers'], pre_model['prior']
        pre_xi = pre_prior.sample_n(1)
        x = ft_flow(pre_layers, pre_xi)
    xi = ft_flow_inv(layers, x)
    f = ft_force(param, layers, xi)
    f_s = torch.linalg.norm(f)
    print(f_s)

# fthmc

ft_hmc_info_list = []

def ft_leapfrog(param, flow, x, p):
    mom_norm = torch.sum(p*p)
    info_list = []
    dt = param.dt
    x_ = x + 0.5*dt*p
    f = ft_force(param, flow, x_)
    p_ = p + (-dt)*f
    info = np.array((float(torch.linalg.norm(f)),
                     float(ft_action(param, flow, x_).detach()),
                     float(torch.sum(p*p_)/np.sqrt(mom_norm*torch.sum(p_*p_)))))
    info_list.append(info)
    for i in range(param.nstep-1):
        x_ = x_ + dt*p_
        f = ft_force(param, flow, x_)
        info = np.array((float(torch.linalg.norm(f)),
                        float(ft_action(param, flow, x_).detach()),
                        float(torch.sum(p*p_)/np.sqrt(mom_norm*torch.sum(p_*p_)))))
        info_list.append(info)
        p_ = p_ + (-dt)*f
    x_ = x_ + 0.5*dt*p_
    print(np.sqrt(average([l[0]**2 for l in info_list])),
          (info_list[0][1], info_list[-1][1]),
          info_list[-1][2])
    ft_hmc_info_list.append(info_list)
    return (x_, p_)

def ft_hmc(param, flow, field):
    x = ft_flow_inv(flow, field)
    p = torch.randn_like(x)
    act0 = ft_action(param, flow, x).detach() + 0.5*torch.sum(p*p)
    x_, p_ = ft_leapfrog(param, flow, x, p)
    xr = regularize(x_)
    act = ft_action(param, flow, xr).detach() + 0.5*torch.sum(p_*p_)
    prob = torch.rand([], dtype=torch.float64)
    dH = act-act0
    exp_mdH = torch.exp(-dH)
    acc = prob < exp_mdH
    # ADJUST ME
    newx = xr if acc else x
    # newx = xr
    newfield = ft_flow(flow, newx)
    return (float(dH), float(exp_mdH), acc, newfield)

def ft_run(param, flow, field = None):
    if field == None:
        field = param.initializer()
    ft_hmc_info_list.clear()
    topo_history.clear()
    fn = param.uniquestr("_ft")
    if os.path.exists(fn):
        return field
    with open(fn, "w") as O:
        params = param.summary()
        O.write(params)
        put(params)
        plaq, topo = (action(param, field) / (-param.beta*param.volume), topocharge(field))
        status = f"Initial configuration:  plaq: {plaq}  topo: {topo} {field.shape}\n"
        O.write(status)
        put(status)
        ts = []
        for n in range(param.nrun):
            t = -timer()
            for i in range(param.ntraj):
                field_run = torch.reshape(field,(1,)+field.shape)
                dH, exp_mdH, acc, field_run = ft_hmc(param, flow, field_run)
                field = field_run[0]
                plaq = action(param, field) / (-param.beta*param.volume)
                topo = topocharge(field)
                ifacc = "ACCEPT" if acc else "REJECT"
                status = f"Traj: {n*param.ntraj+i+1:4}  {ifacc}  dH: {dH:< 12.8}  exp(-dH): {exp_mdH:< 12.8}  plaq: {plaq:< 12.8}  topo: {topo:< 3.3}\n"
                O.write(status)
                put(status)
                topo_history.append(topo.item())
                sys.stdout.flush()
            t += timer()
            ts.append(t)
        print("Run times: ", ts)
        print("Per trajectory: ", [t/param.ntraj for t in ts])
    show_fthmc_stats()
    fn = param.uniquestr("_ft_topo")
    save_topo_change_sqr(fn)
    return field

def show_fthmc_stats():
    fl_list = flattern(ft_hmc_info_list)
    l_len = len(fl_list)
    fl_list = fl_list[l_len // 2:]
    action_list = np.array([l[1] for l in fl_list])
    action_list = sub_avg(action_list)
    print("action sigma", np.sqrt(average(action_list**2)))
    force_list = np.array([l[0] for l in fl_list])
    print("avg force", np.sqrt(average(force_list**2)))

# change size

def get_nets(layers):
    nets = []
    for l in layers:
        nets.append(l.plaq_coupling.net)
    return nets

def make_u1_equiv_layers_net(*, lattice_shape, nets):
    n_layers = len(nets)
    layers = []
    for i in range(n_layers):
        # periodically loop through all arrangements of maskings
        mu = i % 2
        off = (i//2) % 4
        net = nets[i]
        plaq_coupling = NCPPlaqCouplingLayer(
            net, mask_shape=lattice_shape, mask_mu=mu, mask_off=off)
        link_coupling = GaugeEquivCouplingLayer(
            lattice_shape=lattice_shape, mask_mu=mu, mask_off=off, 
            plaq_coupling=plaq_coupling)
        layers.append(link_coupling)
    return torch.nn.ModuleList(layers)

def flow_resize(flow, lat_new):
    # e.g. lat_new = (32, 32)
    return make_u1_equiv_layers_net(lattice_shape = lat_new, nets = get_nets(flow))

# set param

# ADJUST ME

torch.manual_seed(3647)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.set_default_tensor_type(torch.DoubleTensor)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

# ADJUST ME

beta_sim = 5.0

lat_train = (16, 16)

lat_large = (32, 32)

# ADJUST ME

param_run_hmc = Param(
    beta = beta_sim,
    lat = lat_train,
    tau = 1.0,
    nstep = 32,
    ntraj = 128,
    )

param_run_hmc_large = Param(
    beta = beta_sim,
    lat = lat_large,
    tau = 1.0,
    nstep = 32,
    ntraj = 128,
    )

param_train = Param(
    beta = beta_sim,
    lat = lat_train,
    )

param_run_fthmc = Param(
    beta = beta_sim,
    lat = lat_train,
    tau = 1.0,
    nstep = 64,
    ntraj = 128,
    )

param_run_fthmc_large = Param(
    beta = beta_sim,
    lat = lat_large,
    tau = 1.0,
    nstep = 64,
    ntraj = 128,
    )

# run

field = run(param_run_hmc)

field_large = run(param_run_hmc_large)

flow = train_load_save(param_train)

field_fthmc = ft_run(param_run_fthmc, flow)

field_fthm_large = ft_run(param_run_fthmc_large, flow_resize(flow, param_run_fthmc_large.lat))

print("finished")
