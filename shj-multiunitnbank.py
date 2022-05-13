#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:14:01 2021

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools as it
import pickle

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitClusterNBanks import (MultiUnitClusterNBanks, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

# %%

saveplots = 0

six_problems = [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],

                [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],

                ]

# set problem
problem = 5
stim = six_problems[problem]
stim = torch.tensor(stim, dtype=torch.float)
inputs = stim[:, 0:-1]
output = stim[:, -1].long()  # integer


# model details
attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
n_units = 500
# n_dims = inputs.shape[1]
n_dims = 3
loss_type = 'cross_entropy'

# top k%. so .05 = top 5%
k = .05

# n banks of units
n_banks = 2

# SHJ

# trials, etc.
n_epochs = 16

# new local attn - scaling lr
lr_scale = (n_units * k) / 1

# merged - some kept same across banks
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': [1.5, 3.5],  # flips works with this even w same phi! prev [.8, 3.5]
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': [1.5, 1.5],  # can flip work if phi is same - so 2 banks are competiting at the outputs. yes
    'beta': 1,
    'lr_attn': [.25, .002],  # [.15, .02] also works
    'lr_nn': .025/lr_scale,  # scale by n_units*k - keep the same for now
    'lr_clusters': [.01, .01],
    'lr_clusters_group': [.1, .1],
    'k': k
    }

# testing type 1 vs 6
# - flips for type 1
params = {
    'r': 1,
    'c': [1.5, 2.5],  # [1.5, 2.5] / [1.5, 3.5]
    'p': 1,
    'phi': [1.5, 1.5],
    'beta': 1,
    'lr_attn': [.25, .02],  # [.25, .02]
    'lr_nn': .025/lr_scale,
    'lr_clusters': [.01, .01],
    'lr_clusters_group': [.1, .1],
    'k': k
    }


# high c / low c from SHJ testing
# - changing lr_attn and lr_nn, keeping phi constant
params = {
    'r': 1,
    'c': [1., 3.],  # c=.8/1. for type I. c=1. works better for type II.
    'p': 1,
    'phi': [1.5, 1.5],
    'beta': 1,
    'lr_attn': [.35, .002],  # [.25, .02]  # .35 so type 2 wins (if shuffle)
    'lr_nn': [.15/lr_scale, .025/lr_scale],
    'lr_clusters': [.01, .01],
    'lr_clusters_group': [.1, .1],
    'k': k
    }

# # testing - when low c overtakes high c for type 1
# # - attn matters here - this is why type 1 wins. i guess point is if high c
# # and low attn, can't win
# # - can even have it equiv lr_nn
# params = {
#     'r': 1,
#     'c': [.8, 2.5],  # .8/.9
#     'p': 1,
#     'phi': [1.5, 1.5],
#     'beta': 1,
#     'lr_attn': [.15, .002],
#     'lr_nn': [.15/lr_scale, .05/lr_scale],
#     'lr_clusters': [.01, .01],
#     'lr_clusters_group': [.1, .1],
#     'k': k
#     }
# # keeping lr_nn same
# params = {
#     'r': 1,
#     'c': [1.35, 2.5],
#     'p': 1,
#     'phi': [1.5, 1.5],
#     'beta': 1,
#     'lr_attn': [.15, .002],
#     'lr_nn': [.15/lr_scale, .15/lr_scale],
#     'lr_clusters': [.01, .01],
#     'lr_clusters_group': [.1, .1],
#     'k': k
#     }

# testing - when high c overtakes low c for type 6
# - attn doesn't matter here
# - point is that high c should win - same attn, same lr_nn.
# - of course higher lr_nn, low c wins, but that's forcing it
# params = {
#     'r': 1,
#     'c': [1.01, 2.55],
#     'p': 1,
#     'phi': [1.5, 1.5],
#     'beta': 1,
#     'lr_attn': [.15, .15],  # attn doesn't matter here
#     'lr_nn': [.05/lr_scale, .05/lr_scale],
#     'lr_clusters': [.01, .01],
#     'lr_clusters_group': [.1, .1],
#     'k': k
#     }


# testing other types, 2-5
# type 2: c=[1.25, 2.5],lr_attn=[.5, .002], lr_nn=[.15, .05]
# type 3/4: c=[1.25, 2.5],lr_attn=[.5, .002], lr_nn=[.2, .05]
# type 5: ... turns out recruits 4 clus only. lr_nn>.5 then recruits 5!?
# - interaction between the two models is affecting recruitment?
# - need to look further into this. but maybe need recruitment in separate
# banks...
# - OR this is interesting. having 2 banks allows you to solve the problem
# differently. e.g. type 5 with a rulex like representation.

# - hmm, when shuffle, 6 clus is more typical. and sometimes 1+.. CHECK
# - plus, when shuffle, above params for type 3/4 also work


# - seems attn doesn't matter much.. main lr_nn. 
# i guess this might be because type I is only one when attn really drives acc
# since in other cases, placing a cluster on them does a lot...?
# params = {
#     'r': 1,
#     'c': [1.25, 2.5],
#     'p': 1,
#     'phi': [1.5, 1.5],
#     'beta': 1,
#     'lr_attn': [.005, .005],
#     'lr_nn': [.2/lr_scale, .05/lr_scale],
#     'lr_clusters': [.01, .01],
#     'lr_clusters_group': [.1, .1],
#     'k': k
#     }

# new after fixed phi
params = {
    'r': 1,
    'c': [.75, 2.5],  # c=.8/1. for type I. c=1. works better for II.
    'p': 1,
    'phi': [1.3, 1.2],  # 1.2/1.1 for latter atm
    'beta': 1,
    'lr_attn': [.2, .002],  # [.25, .02]
    'lr_nn': [.05/lr_scale, .01/lr_scale],  # latter also tried .0075, not as gd tho
    'lr_clusters': [.05, .05],
    'lr_clusters_group': [.1, .1],
    'k': k
    }

model = MultiUnitClusterNBanks(n_units, n_dims, n_banks, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shj_order=False)

# pr target
plt.plot(1 - epoch_ptarget.T.detach())
plt.ylim([0, .5])
plt.title('Type {}'.format(problem+1))
plt.gca().legend(('total output',
                  'c = {}'.format(model.params['c'][0]),
                  'c = {}'.format(model.params['c'][1])
                  ))

if saveplots:
    p = [model.params['c'][0], model.params['c'][1],
         model.params['lr_attn'][0], model.params['lr_attn'][1],
         model.params['lr_nn'][0], model.params['lr_nn'][1]]

    figname = os.path.join(figdir,
                           'nbanks_SHJ_type{}_c{}_{}_attn{}_{}_nn{}_{}'.format(
                               problem+1, p[0], p[1], p[2], p[3], p[4], p[5]))
    plt.savefig(figname + '.png', dpi=100)
plt.show()


# # attention weights
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(torch.stack(model.attn_trace, dim=0)[:, :, 0])
# ax[0].set_ylim([torch.stack(model.attn_trace, dim=0).min()-.01,
#                 torch.stack(model.attn_trace, dim=0).max()+.01])
# ax[1].plot(torch.stack(model.attn_trace, dim=0)[:, :, 1])
# ax[1].set_ylim([torch.stack(model.attn_trace, dim=0).min()-.01,
#                 torch.stack(model.attn_trace, dim=0).max()+.01])
# plt.show()

# %% SHJ

saveplots = False

saveresults = True

niter = 1

n_banks = 2

six_problems = [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

                [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
                 [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],

                [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],

                ]


n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
pt_all = torch.zeros([niter, 6, n_banks+1, n_epochs])

# model details
attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
n_units = 2000
loss_type = 'cross_entropy'
k = .01
lr_scale = (n_units * k)


rec_all =[[] for i in range(6)]
nrec_all = torch.zeros([niter, 6, n_banks])

# fc1 ws - list since diff ntrials (since it appends when recruit & upd)
w_trace = [[] for i in range(6)]
act_trace = [[] for i in range(6)]
attn_trace = [[] for i in range(6)]

# run multiple iterations
for i in range(niter):

    # six problems
    for problem in range(6):  # [0, 5]: #  np.array([4]):

        stim = six_problems[problem]
        stim = torch.tensor(stim, dtype=torch.float)
        inputs = stim[:, 0:-1]
        output = stim[:, -1].long()  # integer

        # 16 per trial
        inputs = inputs.repeat(2, 1)
        output = output.repeat(2).T
        n_dims = inputs.shape[1]

        # gridsearch + fgsearch
        # tensor([[6.0000e-01, 1.0000e+00, 1.5500e+00, 7.0000e-01, 3.0000e-01, 8.0000e-01,
#                  1.7000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
        params = {
            'r': 1,
            'c': [.6, 1.7],
            'p': 1,
            'phi': [1., 2.25],
            'beta': 1,
            'lr_attn': [1.55, .001],
            'lr_nn': [.7/lr_scale, .01/lr_scale],
            'lr_clusters': [.3, .3],
            'lr_clusters_group': [.8, .5],
            'k': k
            }

        # gsearch + finegsearch 2022
        # tensor([[7.0000e-01, 8.7500e-01, 1.3000e+00, 8.0000e-01, 3.0000e-01, 5.0000e-01,
        #          1.7000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
        params = {
            'r': 1,
            'c': [.7, 1.7],
            'p': 1,
            'phi': [.875, 2.25],
            'beta': 1,
            'lr_attn': [1.3, .001],
            'lr_nn': [.8/lr_scale, .01/lr_scale],
            'lr_clusters': [.3, .3],
            'lr_clusters_group': [.5, .5],
            'k': k
            }

        model = MultiUnitClusterNBanks(n_units, n_dims, n_banks, attn_type, k,
                                       params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle_seed=1, shj_order=False)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        # don't save for big sim
        w_trace[problem].append(torch.stack(model.fc1_w_trace))
        act_trace[problem].append(torch.stack(model.fc1_act_trace))
        attn_trace[problem].append(torch.stack(model.attn_trace))

        # save n clusters recruited
        rec_all[problem].append(model.recruit_units_trl)  # saves the seq
        # nclus recruited
        nrec_all[i, problem] = torch.tensor([len(model.recruit_units_trl[0]),
                                             len(model.recruit_units_trl[1])])

        print(model.recruit_units_trl)
        # print(model.recruit_units_trl[0] == model.recruit_units_trl[1])
        # print(np.unique(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=1), axis=0))


# save variables
# - pt_all, nrec_all
if saveresults:
    fn = os.path.join(datadir,
                      'shj_nbanks_results_pt_nrec_k{}_{}units.pkl'.format(
                          k, n_units))
    
    shj_res = [pt_all, nrec_all]  # seeds_all
    open_file = open(fn, "wb")
    pickle.dump(shj_res, open_file)
    open_file.close()


import matplotlib.font_manager as font_manager
# for roman numerals
font = font_manager.FontProperties(family='Tahoma',
                                   style='normal', size=7)

fntsiz = 15
ylims = (0., .55)

fig, ax = plt.subplots(1, 3)
ax[0].plot(pt_all[:, :, 0].mean(axis=0).T)
ax[0].set_ylim(ylims)
ax[1].plot(pt_all[:, :, 1].mean(axis=0).T)
ax[1].set_ylim(ylims)
ax[2].plot(pt_all[:, :, 2].mean(axis=0).T)
ax[2].set_ylim(ylims)
ax[2].legend(('I', 'II', 'III', 'IV', 'V', 'VI'), fontsize=10)
plt.tight_layout()
if saveplots:
    figname = (
        os.path.join(figdir,
                     'shj_nbanks{}_curves_k{}_{}units_{}sims.pdf'.format(
                         n_banks, k, n_units, niter))
    )
    plt.savefig(figname)
plt.show()

# plot out
fig, ax = plt.subplots(1, 1)
ax.plot(pt_all[:, :, 0].mean(axis=0).T)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.set_title('Full model output', fontsize=fntsiz)
plt.tight_layout()
if saveplots:
    figname = (
        os.path.join(figdir,
                     'shj_nbanks{}_curves_sep_b12_c{}{}_k{}_{}units_{}sims.pdf'
                     .format(n_banks, params['c'][0], params['c'][1], k,
                             n_units, niter))
    )
    plt.savefig(figname)
plt.show()

# plot single to compare with single bank
# bank 1
fig, ax = plt.subplots(1, 1)
ax.plot(pt_all[:, :, 1].mean(axis=0).T)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.set_title('bank 1: c = {}'.format(params['c'][0]), fontsize=fntsiz)
if saveplots:
    figname = (
        os.path.join(figdir,
                     'shj_nbanks{}_curves_sep_b1_c{}_k{}_{}units_{}sims.pdf'
                     .format(n_banks, params['c'][0], k, n_units, niter))
    )
    plt.savefig(figname)
plt.show()

# bank 2
fig, ax = plt.subplots(1, 1)
ax.plot(pt_all[:, :, 2].mean(axis=0).T)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.set_title('bank 2: c = {}'.format(params['c'][1]), fontsize=fntsiz)
if saveplots:
    figname = (
        os.path.join(figdir,
                     'shj_nbanks{}_curves_sep_b2_c{}_k{}_{}units_{}sims.pdf'
                     .format(n_banks, params['c'][1], k, n_units, niter))
    )
    plt.savefig(figname)
plt.show()

# %%

saveplots = False
# have to go through each iteration, since different ntrials if diff n recruit

# for i in range(niter):
# print(w_trace[problem][i])

i = 0
problem = 5
act = act_trace[problem][i]

# output (pr / activations with phi)
ylims = (0, 1)

fig, ax = plt.subplots(1, 2)
ax[0].plot(act[:, 1])
ax[0].set_title('type {}, c = {}'.format(problem+1, params['c'][0]))
ax[0].set_ylim(ylims)
ax[1].plot(act[:, 2])
ax[1].set_title('type {}, c = {}'.format(problem+1, params['c'][1]))
ax[1].set_ylim(ylims)

if saveplots:
    figname = (
        os.path.join(figdir, 'shj_nbanks{}_act_type{}_k{}_{}units.pdf'.format(
            problem+1, n_banks, k, n_units))
    )
    plt.savefig(figname)
plt.show()

# change in activation magnitude over time
# - if want do do this by block, need to only get activations / weights after
# recruit. complication now since one bank can recruit and the other not.
# - in a way, including recruit is fine too (since at forward func), sliding
# win more flexible.


def sliding_window(iterable, n):
    iterables = it.tee(iterable, n)
    for iterable, num_skipped in zip(iterables, it.count()):
        for _ in range(num_skipped):
            next(iterable, None)
    return np.array(list((zip(*iterables))))


winsize = 16  # ntrials to compute running average

t1 = sliding_window(act[:, 1, 0], winsize)  # just 1 of the outputs
t2 = sliding_window(act[:, 2, 0], winsize)

fig, ax = plt.subplots(1, 2)
ax[0].plot(np.diff(t1))
ax[0].set_title('type {}, c = {}'.format(problem+1, params['c'][0]))
ax[0].set_ylim(ylims)
ax[1].plot(np.diff(t2))
ax[1].set_title('type {}, c = {}'.format(problem+1, params['c'][1]))
ax[1].set_ylim(ylims)
if saveplots:
    figname = (
        os.path.join(figdir, 'shj_nbanks{}_actdiff_type{}_k{}_{}units.pdf'.format(
            problem+1, n_banks, k, n_units))
    )
    plt.savefig(figname)
plt.show()


# weights
for problem in range(6):
    w = w_trace[problem][i]
    
    # ylims = (-torch.max(torch.abs(w)), torch.max(torch.abs(w)))
    ylims = (-.06, .06)
    
    w0 = w[:, :, model.bmask[0]]
    w0 = torch.reshape(w0, (w0.shape[0], w0.shape[1] * w0.shape[2]))
    plt.plot(w0[:, torch.nonzero(w0.sum(axis=0)).squeeze()])
    plt.ylim(ylims)
    plt.title('assoc ws, type {}, c = {}'.format(problem+1, params['c'][0]))
    if saveplots:
        figname = (
            os.path.join(figdir,
                          'shj_nbanks{}_assocw_sep_type{}_c{}_k{}_{}units.pdf'.format(
                              n_banks, problem+1, params['c'][0], k, n_units))
        )
        plt.savefig(figname)
    plt.show()
    
    w1 = w[:, :, model.bmask[1]]
    w1 = torch.reshape(w1, (w1.shape[0], w1.shape[1] * w1.shape[2]))
    plt.plot(w1[:, torch.nonzero(w1.sum(axis=0)).squeeze()])
    plt.ylim(ylims)
    plt.title('assoc ws, type {}, c = {}'.format(problem+1, params['c'][1]))
    if saveplots:
        figname = (
            os.path.join(figdir,
                          'shj_nbanks{}_assocw_sep_type{}_c{}_k{}_{}units.pdf'.format(
                              n_banks, problem+1, params['c'][1], k, n_units))
        )
        plt.savefig(figname)
    plt.show()


# # weight change over time
# winsize = 16  # ntrials to compute running average

# t1 = sliding_window(torch.sum(w0.abs(), dim=1), winsize)
# t2 = sliding_window(torch.sum(w1.abs(), dim=1), winsize)

# ylims = (0, torch.max(torch.tensor([np.diff(t1), np.diff(t2)])) + .01)

# plt.plot(np.diff(t1))
# plt.ylim(ylims)
# plt.show()
# plt.plot(np.diff(t2))
# plt.ylim(ylims)
# plt.show()
