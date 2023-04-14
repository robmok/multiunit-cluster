#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:44:37 2021

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools as it

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

# matplotlib first 6 default colours
col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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

# %% lesioning experiments

saveplots = False

n_sims = 1

problem = 0

stim = six_problems[problem]
stim = torch.tensor(stim, dtype=torch.float)
inputs = stim[:, 0:-1]
output = stim[:, -1].long()  # integer
# 16 per trial
inputs = inputs.repeat(2, 1)
output = output.repeat(2).T

# model details
attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
n_dims = inputs.shape[1]
loss_type = 'cross_entropy'

n_epochs = 16

# lesions = {
#     'n_lesions': 10,  # n_lesions per event
#     'gen_rand_lesions_trials': False,  # generate lesion events at random times
#     'pr_lesion_trials': .01,  # if True, set this
#     'lesion_trials': torch.tensor([20])  # if False, set lesion trials
#     }

# for All: need 1 simulation with lesions vs no lesions - w same shuffled seq
# - feed in a random number for seed: shuffle_seed = torch.randperm(n_sims)
# or torch.randperm(n_sims*5)[:n_sims] to get more nums so diff over other sims
# - HMMM you might also want the same shuffle over a set of sims, if randomly
# lesioning units or random time points!


# expt 1: n_lesions [single lesionevent] - number of units. and timing of event
# - manipulation n_lesions
# - manpulate k value and n_total units. will be affects by k most, but of course n_total interacts
# - fix / manipulate: shuffle - seed the same num across a set of sims, then
# seed another num for another set; run nset sims. this is to test same shuffle
# different lesions (since they are random which units get lesioned)
# - fix: lesion_trials at 1 time point (across a few sims, different time pt) 
# save for each sim: model.recruit_units_trl, len(model.recruit_units_trl),
# epoch_ptarget.detach(), model.attn_trace

# expt 2: n lesion events, and timing
# - first, do nlesions early, middle, late. then also do random.
# e.g. [0:10 early, 0 mid, 0 late], then [0 early, 0:10 mid, 0 late], etc.

shuffle_seeds = torch.randperm(n_sims*5)[:n_sims]

# things to manipulate
#  - with 5000/8000 recovers - actually even better (recruit extra cluster so
# higher act... feature/bug? could be feature: learning, hpc synpase overturn)
n_units = [20, 500, 1000, 10000]  # [20, 100, 500]
k = [.05]
n_lesions = [0, 25, 50]
lesion_trials = np.array([[60]])  # [60]]  # 1 per lesion, but do at diff times

# n_units = 100
# n_lesions = 50


sim_ps = []
pt = []
recruit_trial = []
attn_trace = []

# can add loop for problem in range(6)

for sim_prms in it.product(n_units, k, lesion_trials, n_lesions):
    for isim in range(n_sims):

        sim_ps.append(sim_prms)

        # shj params
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': 1.,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 12.5,
            'beta': 1.,
            'lr_attn': .15,  # this scales at grad computation now
            'lr_nn': .015/(sim_prms[0] * sim_prms[1]),  # scale by n_units*k
            'lr_clusters': .01,
            'lr_clusters_group': .1,
            'k': sim_prms[1],
            }

        # from above shj i used
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .7,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 9.,
            'beta': 1.,
            'lr_attn': .35,  # this scales at grad computation now
            'lr_nn': .0075/(sim_prms[0] * sim_prms[1]),
            'lr_clusters': .075,  # .075/.1
            'lr_clusters_group': .12,
            'k': sim_prms[1]
            }

        # or gridsearch params (1st one)
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': 1.6,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1.,
            'beta': 1.,
            'lr_attn': .455,  # this scales at grad computation now
            'lr_nn': .205/(sim_prms[0] * sim_prms[1]),
            'lr_clusters': .305,  # .075/.1
            'lr_clusters_group': .7,
            'k': sim_prms[1]
            }

        # gridsearch params (2nd one)
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .8,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1.,
            'beta': 1.,
            'lr_attn': .8,  # this scales at grad computation now
            'lr_nn': .65/(sim_prms[0] * sim_prms[1]),
            'lr_clusters': .5,  # .075/.1
            'lr_clusters_group': .5,
            'k': sim_prms[1]
            }

        # 2022 gsearch results - saved with _v2 at the end
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .2,
            'p': 1,
            'phi': 14.,
            'beta': 1.,
            'lr_attn': .275,  # /(n_units*k), # 3., # maybe should scale here..!
            'lr_nn': .05/(sim_prms[0] * sim_prms[1]),  # lr_scale
            'lr_clusters': .35,
            'lr_clusters_group':.9,
            'k': sim_prms[1]
            }

        model = MultiUnitCluster(sim_prms[0], n_dims, attn_type, sim_prms[1],
                                 params=params)

        lesions = {
            'n_lesions': sim_prms[3],  # n_lesions per event
            'gen_rand_lesions_trials': False,  # lesion events at random times
            'pr_lesion_trials': .01,  # if True, set this
            'lesion_trials': torch.tensor(sim_prms[2])  # if False, set this
            }

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle_seed=shuffle_seeds[isim],
            lesions=lesions, shj_order=False)

        pt.append(1 - epoch_ptarget.detach())
        recruit_trial.append(model.recruit_units_trl)
        attn_trace.append(torch.stack(model.attn_trace, dim=0))

# % plot
# dotted lines: (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt
# space) with no offset.

fntsiz = 18

# matplotlib first 6 default colours
col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

plt.rcdefaults()

# index to average over sims
ind_sims = [torch.arange(i * n_sims, (i + 1) * n_sims)
            for i in range(len(pt) // n_sims)]

# pt
pts = torch.stack(pt)

# average over sims and plot
# - specify sims by how sim_prms are ordered. so 'range' is indexing n_units,
# plotting by n_lesions
len_p = len(n_lesions)

ylims = (0., 0.55)

fig, ax, = plt.subplots(1, 4)
# 20 units
pt_plot = torch.stack([pts[ind_sims[i]].mean(axis=0) for i in range(0, len_p)])
# fig, ax = plt.subplots(1, 1)
ax[0].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[0].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[0].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[0].set_ylim(ylims)
ax[0].tick_params(axis='x', labelsize=fntsiz-5)
ax[0].tick_params(axis='y', labelsize=fntsiz-5)
# ax[0].legend(('{} lesions'.format(n_lesions[0]),
#            '{} lesions'.format(n_lesions[1]),
#            '{} lesions'.format(n_lesions[2])), fontsize=fntsiz-2)
# ax[0].set_xlabel('Block', fontsize=fntsiz)
ax[0].set_ylabel('Pr of Error', fontsize=fntsiz-3)
ax[0].set_title('{} units'.format(n_units[0]), fontsize=fntsiz-5)
ax[0].set_box_aspect(1)
# plt.tight_layout()
# if saveplots:
#     figname = os.path.join(figdir,
#                            'lesion_pt_type{}_trl{}_{}units_{}sims'.format(
#                                problem+1, lesion_trials[0, 0], n_units[0],
#                                n_sims))
#     plt.savefig(figname, dpi=100)
# plt.show()

pt_plot = [np.nanmean(pts[ind_sims[i]], axis=0) for i in range(len_p, len_p*2)]
ax[1].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[1].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[1].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[1].set_ylim(ylims)
ax[1].tick_params(axis='x', labelsize=fntsiz-5)
ax[1].set_yticklabels([])  # remove ticklables
ax[1].set_xlabel('                Block', fontsize=fntsiz-3)
ax[1].set_title('{} units'.format(n_units[1]), fontsize=fntsiz-5)
ax[1].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*2, len_p*3)]
ax[2].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[2].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[2].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[2].set_ylim(ylims)
ax[2].tick_params(axis='x', labelsize=fntsiz-5)
ax[2].set_yticklabels([])  # remove ticklables
ax[2].set_box_aspect(1)
ax[2].set_title('{} units'.format(n_units[2]), fontsize=fntsiz-5)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*3, len_p*4)]
ax[3].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[3].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[3].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[3].set_ylim(ylims)
ax[3].tick_params(axis='x', labelsize=fntsiz-5)
ax[3].set_yticklabels([])  # remove ticklables
ax[3].set_title('{} units'.format(n_units[3]), fontsize=fntsiz-5)
# ax[3].legend(('{} lesions'.format(n_lesions[0]),
#               '{} lesions'.format(n_lesions[1]),
#               '{} lesions'.format(n_lesions[2])),
#              fontsize=fntsiz-8, bbox_to_anchor=(1.1, 1.), loc="lower left")
ax[3].set_box_aspect(1)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_pt_subplot_k{}_type{}_trl{}_{}-{}-{}-{}'
                           'units_{}sims_v2'.format(
                               sim_prms[1], problem+1, lesion_trials[0, 0],
                               n_units[0],  n_units[1], n_units[2], n_units[3],
                               n_sims))
    plt.savefig(figname + '.png', dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()

# # recruit clusters
# # plt.style.use('seaborn-darkgrid')
# recr_n = torch.tensor(
#     [len(recruit_trial[i]) for i in range(len(recruit_trial))],  # count
#     dtype=torch.float)

# recr_avgs = torch.tensor(
#     [[recr_n[ind_sims[i]].mode() for i in range(0, len_p)],
#      [recr_n[ind_sims[i]].mode() for i in range(len_p, len_p*2)],
#      [recr_n[ind_sims[i]].mode() for i in range(len_p*2, len_p*3)],
#      [recr_n[ind_sims[i]].mode() for i in range(len_p*3, len_p*4)]])

# ylims = (0, recr_avgs[:, :, 0].max() + .5)  # index since mode gives indices..

# mrksiz = 4

# fig, ax, = plt.subplots(1, 4)
# recr_plot = torch.stack(
#     [recr_n[ind_sims[i]].mode().values for i in range(0, len_p)])
# ax[0].plot(['0', '10', '20'], recr_plot, 'o--', color=col[problem],
#            markersize=mrksiz)
# ax[0].set_title('{} units'.format(n_units[0]), fontsize=fntsiz-5)
# ax[0].tick_params(axis='x', labelsize=fntsiz-5)
# ax[0].tick_params(axis='y', labelsize=fntsiz-5)
# ax[0].set_ylabel('No. of recruitments', fontsize=fntsiz-3)
# ax[0].set_ylim(ylims)
# ax[0].set_box_aspect(1)

# recr_plot = torch.stack(
#     [recr_n[ind_sims[i]].mode().values for i in range(len_p, len_p*2)])
# ax[1].plot(['0', '10', '20'], recr_plot, 'o--', color=col[problem],
#            markersize=mrksiz)
# ax[1].set_title('{} units'.format(n_units[1]), fontsize=fntsiz-5)
# ax[1].tick_params(axis='x', labelsize=fntsiz-5)
# ax[1].set_yticklabels([])  # remove ticklables
# ax[1].set_ylim(ylims)
# ax[1].set_xlabel('                    No. of lesions', fontsize=fntsiz-3)
# ax[1].set_box_aspect(1)

# recr_plot = torch.stack(
#     [recr_n[ind_sims[i]].mode().values for i in range(len_p*2, len_p*3)])
# ax[2].plot(['0', '10', '20'], recr_plot, 'o--',
#            color=col[problem], markersize=mrksiz)
# ax[2].set_title('{} units'.format(n_units[2]), fontsize=fntsiz-5)
# ax[2].tick_params(axis='x', labelsize=fntsiz-5)
# ax[2].set_yticklabels([])  # remove ticklables
# ax[2].set_ylim(ylims)
# ax[2].set_box_aspect(1)

# recr_plot = torch.stack(
#     [recr_n[ind_sims[i]].mode().values for i in range(len_p*3, len_p*4)])
# ax[3].plot(['0', '10', '20'], recr_plot, 'o--', color=col[problem],
#            markersize=mrksiz)
# ax[3].set_title('{} units'.format(n_units[3]), fontsize=fntsiz-5)
# ax[3].tick_params(axis='x', labelsize=fntsiz-5)
# ax[3].set_yticklabels([])  # remove ticklables
# ax[3].set_ylim(ylims)
# ax[3].set_box_aspect(1)
# plt.tight_layout()

# if saveplots:
#     figname = os.path.join(figdir,
#                            'lesion_recruit_k{}_type{}_trl{}_{}-{}-{}-{}units_'
#                            '{}sims'.format(
#                                sim_prms[1], problem+1, lesion_trials[0, 0],
#                                n_units[0], n_units[1], n_units[2], n_units[3],
#                                n_sims))
#     plt.savefig(figname + '.png', dpi=100)
#     plt.savefig(figname + '.pdf')
# plt.show()


# # back to defaults
# plt.rcdefaults()

# attn
# # - are these interpretable if averaged over? probably not for some problems
# attns = torch.stack(attn_trace)

# ylims = (attns.min() - .01, attns.max() + .01)

# # 20 units
# attn_plot = torch.stack(
#     [attns[ind_sims[i]].mean(axis=0) for i in range(0, len_p)])
# fig, ax = plt.subplots(1, 3)
# for iplt in range(len_p):
#     ax[iplt].plot(attn_plot[iplt])
#     ax[iplt].set_ylim(ylims)
#     ax[iplt].set_title('{} units, {} lesions'.format(n_units[0],
#                                                      n_lesions[iplt]),
#                        fontsize=10)
# if saveplots:
#     figname = os.path.join(figdir,
#                            'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
#                                problem+1, lesion_trials[0, 0], n_units[0],
#                                n_sims))
#     plt.savefig(figname, dpi=100)
# plt.show()

# # 100 units
# attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(len_p, len_p*2)]
# fig, ax = plt.subplots(1, 3)
# for iplt in range(len_p):
#     ax[iplt].plot(torch.stack(attn_plot)[iplt])
#     ax[iplt].set_ylim(ylims)
#     ax[iplt].set_title('{} units, {} lesions'.format(n_units[1],
#                                                      n_lesions[iplt]),
#                        fontsize=10)
# if saveplots:
#     figname = os.path.join(figdir,
#                            'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
#                                problem+1, lesion_trials[0, 0], n_units[1],
#                                n_sims))
#     plt.savefig(figname, dpi=100)
# plt.show()

# # 500 units
# attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(len_p*2, len_p*3)]
# fig, ax = plt.subplots(1, 3)
# for iplt in range(len_p):
#     ax[iplt].plot(torch.stack(attn_plot)[iplt])
#     ax[iplt].set_ylim(ylims)
#     ax[iplt].set_title('{} units, {} lesions'.format(n_units[2],
#                                                      n_lesions[iplt]),
#                        fontsize=10)
# if saveplots:
#     figname = os.path.join(figdir,
#                            'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
#                                problem+1, lesion_trials[0, 0], n_units[2],
#                                n_sims))
#     plt.savefig(figname, dpi=100)
# plt.show()

# attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(len_p*3, len_p*4)]
# fig, ax = plt.subplots(1, 3)
# for iplt in range(len_p):
#     ax[iplt].plot(torch.stack(attn_plot)[iplt])
#     ax[iplt].set_ylim(ylims)
#     ax[iplt].set_title('{} units, {} lesions'.format(n_units[3],
#                                                      n_lesions[iplt]),
#                        fontsize=10)
# if saveplots:
#     figname = os.path.join(figdir,
#                            'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
#                                problem+1, lesion_trials[0, 0], n_units[3],
#                                n_sims))
#     plt.savefig(figname, dpi=100)
# plt.show()



# For plotting, make df?

# import pandas as pd
# df_sum = pd.DataFrame(columns=['acc', 'k', 'n_uni'ts, 'n_lesions', 'lesion trials', 'sim_num'])
