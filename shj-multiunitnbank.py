#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:14:01 2021

@author: robert.mok
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import pickle

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitClusterNBanks import (MultiUnitClusterNBanks, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

# %% SHJ 6 problems

saveplots = False

saveresults = False

niter = 1  # 25

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

# %% exploring  change in weights over learning

# saveplots = False
# # have to go through each iteration, since different ntrials if diff n recruit

# # for i in range(niter):
# # print(w_trace[problem][i])

# i = 0
# problem = 5
# act = act_trace[problem][i]

# # output (pr / activations with phi)
# ylims = (0, 1)

# fig, ax = plt.subplots(1, 2)
# ax[0].plot(act[:, 1])
# ax[0].set_title('type {}, c = {}'.format(problem+1, params['c'][0]))
# ax[0].set_ylim(ylims)
# ax[1].plot(act[:, 2])
# ax[1].set_title('type {}, c = {}'.format(problem+1, params['c'][1]))
# ax[1].set_ylim(ylims)

# if saveplots:
#     figname = (
#         os.path.join(figdir, 'shj_nbanks{}_act_type{}_k{}_{}units.pdf'.format(
#             problem+1, n_banks, k, n_units))
#     )
#     plt.savefig(figname)
# plt.show()

# # change in activation magnitude over time
# # - if want do do this by block, need to only get activations / weights after
# # recruit. complication now since one bank can recruit and the other not.
# # - in a way, including recruit is fine too (since at forward func), sliding
# # win more flexible.


# def sliding_window(iterable, n):
#     iterables = it.tee(iterable, n)
#     for iterable, num_skipped in zip(iterables, it.count()):
#         for _ in range(num_skipped):
#             next(iterable, None)
#     return np.array(list((zip(*iterables))))


# winsize = 16  # ntrials to compute running average

# t1 = sliding_window(act[:, 1, 0], winsize)  # just 1 of the outputs
# t2 = sliding_window(act[:, 2, 0], winsize)

# fig, ax = plt.subplots(1, 2)
# ax[0].plot(np.diff(t1))
# ax[0].set_title('type {}, c = {}'.format(problem+1, params['c'][0]))
# ax[0].set_ylim(ylims)
# ax[1].plot(np.diff(t2))
# ax[1].set_title('type {}, c = {}'.format(problem+1, params['c'][1]))
# ax[1].set_ylim(ylims)
# if saveplots:
#     figname = (
#         os.path.join(figdir, 'shj_nbanks{}_actdiff_type{}_k{}_{}units.pdf'.format(
#             problem+1, n_banks, k, n_units))
#     )
#     plt.savefig(figname)
# plt.show()


# # weights
# for problem in range(6):
#     w = w_trace[problem][i]
    
#     # ylims = (-torch.max(torch.abs(w)), torch.max(torch.abs(w)))
#     ylims = (-.06, .06)
    
#     w0 = w[:, :, model.bmask[0]]
#     w0 = torch.reshape(w0, (w0.shape[0], w0.shape[1] * w0.shape[2]))
#     plt.plot(w0[:, torch.nonzero(w0.sum(axis=0)).squeeze()])
#     plt.ylim(ylims)
#     plt.title('assoc ws, type {}, c = {}'.format(problem+1, params['c'][0]))
#     if saveplots:
#         figname = (
#             os.path.join(figdir,
#                           'shj_nbanks{}_assocw_sep_type{}_c{}_k{}_{}units.pdf'.format(
#                               n_banks, problem+1, params['c'][0], k, n_units))
#         )
#         plt.savefig(figname)
#     plt.show()
    
#     w1 = w[:, :, model.bmask[1]]
#     w1 = torch.reshape(w1, (w1.shape[0], w1.shape[1] * w1.shape[2]))
#     plt.plot(w1[:, torch.nonzero(w1.sum(axis=0)).squeeze()])
#     plt.ylim(ylims)
#     plt.title('assoc ws, type {}, c = {}'.format(problem+1, params['c'][1]))
#     if saveplots:
#         figname = (
#             os.path.join(figdir,
#                           'shj_nbanks{}_assocw_sep_type{}_c{}_k{}_{}units.pdf'.format(
#                               n_banks, problem+1, params['c'][1], k, n_units))
#         )
#         plt.savefig(figname)
#     plt.show()


# # # weight change over time
# # winsize = 16  # ntrials to compute running average

# # t1 = sliding_window(torch.sum(w0.abs(), dim=1), winsize)
# # t2 = sliding_window(torch.sum(w1.abs(), dim=1), winsize)

# # ylims = (0, torch.max(torch.tensor([np.diff(t1), np.diff(t2)])) + .01)

# # plt.plot(np.diff(t1))
# # plt.ylim(ylims)
# # plt.show()
# # plt.plot(np.diff(t2))
# # plt.ylim(ylims)
# # plt.show()
