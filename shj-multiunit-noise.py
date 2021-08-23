#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:47:02 2021

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

# %% noise experiments

saveplots = False

n_sims = 5

problem = 4

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

shuffle_seeds = torch.randperm(n_sims*5)[:n_sims]

# things to manipulate
n_units = [20, 200, 2000, 20000]  # [20, 100, 1000, 10000]
k = [.05]
lr_group = [0., .3, .6, .9]
noise_upd1 = [0., .6, 1.2]

sim_ps = []
pt = []
recruit_trial = []
attn_trace = []

# can add loop for problem in range(6)
for s_cnt, sim_prms in enumerate(it.product(n_units, k, lr_group, noise_upd1)):
    print('Running {} / {} param sets'.format(
        s_cnt+1, len(list(it.product(n_units, k, lr_group, noise_upd1)))))
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
            'lr_clusters_group': sim_prms[2],
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
            'lr_clusters_group': sim_prms[2],
            'k': sim_prms[1]
            }

        # # or gridsearch params (1st one)
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 1.6,  # w/ attn grad normalized, c can be large now
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 1.,
        #     'beta': 1.,
        #     'lr_attn': .455,  # this scales at grad computation now
        #     'lr_nn': .205/(sim_prms[0] * sim_prms[1]),
        #     'lr_clusters': .305,  # .075/.1
        #     'lr_clusters_group':  sim_prms[2],
        #     'k': sim_prms[1]
        #     }

        # # gridsearch params (2nd one)
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .8,  # w/ attn grad normalized, c can be large now
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 1.,
        #     'beta': 1.,
        #     'lr_attn': .8,  # this scales at grad computation now
        #     'lr_nn': .65/(sim_prms[0] * sim_prms[1]),
        #     'lr_clusters': .5,  # .075/.1
        #     'lr_clusters_group': sim_prms[2],
        #     'k': sim_prms[1]
        #     }

        # noise - mean and sd of noise to be added
        noise = {'update1': [0, sim_prms[3]],
                 'update2': [0, .0],  # no noise here also makes sense
                 'recruit': [0., .0]}

        model = MultiUnitCluster(sim_prms[0], n_dims, attn_type, sim_prms[1],
                                 params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle=True, noise=noise,
            shuffle_seed=shuffle_seeds[isim], shj_order=True)

        pt.append(1 - epoch_ptarget.detach())
        recruit_trial.append(model.recruit_units_trl)
        attn_trace.append(torch.stack(model.attn_trace, dim=0))

# %% plot
# dotted lines: (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt
# space) with no offset.

fntsiz = 18

# matplotlib first 6 default colours
col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

plt.rcdefaults()

# index to average over sims
ind_sims = [torch.arange(i * n_sims, (i + 1) * n_sims)
            for i in range(len(sim_ps) // n_sims)]

# pt
pts = torch.stack(pt)

# average over sims and plot
# - specify sims by how sim_prms are ordered. so 'range' is indexing n_units,
# plotting each lr_group, by noise_upd1 conditions
len_p = len(noise_upd1)

ylims = (0., 0.55)

# 20 units
fig, ax, = plt.subplots(1, 4)
pt_plot = torch.stack([pts[ind_sims[i]].mean(axis=0) for i in range(0, len_p)])
ax[0].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[0].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[0].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[0].set_ylim(ylims)
ax[0].tick_params(axis='x', labelsize=fntsiz-5)
ax[0].tick_params(axis='y', labelsize=fntsiz-5)
ax[0].set_ylabel('Pr of Error', fontsize=fntsiz-3)
ax[0].set_title('{}, grp lr {}'.format(n_units[0], lr_group[0]),
                fontsize=fntsiz-8)
ax[0].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p, len_p*2)]
ax[1].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[1].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[1].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[1].set_ylim(ylims)
ax[1].tick_params(axis='x', labelsize=fntsiz-5)
ax[1].set_yticklabels([])  # remove ticklables
ax[1].set_xlabel('                Block', fontsize=fntsiz-3)
ax[1].set_title('{}, grp lr {}'.format(n_units[0], lr_group[1]),
                fontsize=fntsiz-8)
ax[1].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*2, len_p*3)]
ax[2].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[2].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[2].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[2].set_ylim(ylims)
ax[2].tick_params(axis='x', labelsize=fntsiz-5)
ax[2].set_yticklabels([])  # remove ticklables
ax[2].set_box_aspect(1)
ax[2].set_title('{}, grp lr {}'.format(n_units[0], lr_group[2]),
                fontsize=fntsiz-8)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*3, len_p*4)]
ax[3].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[3].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[3].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[3].set_ylim(ylims)
ax[3].tick_params(axis='x', labelsize=fntsiz-5)
ax[3].set_yticklabels([])  # remove ticklables
ax[3].set_title('{}, grp lr {}'.format(n_units[0], lr_group[3]),
                fontsize=fntsiz-8)
# ax[3].legend(('{} lesions'.format(n_lesions[0]),
#               '{} lesions'.format(n_lesions[1]),
#               '{} lesions'.format(n_lesions[2])),
#              fontsize=fntsiz-8, bbox_to_anchor=(1.1, 1.), loc="lower left")
ax[3].set_box_aspect(1)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'noise_pt_subplot_k{}_type{}_{}units_noise{}-{}-{}'
                           '_{}sims'.format(
                                sim_prms[1], problem+1, n_units[0],
                                noise_upd1[0], noise_upd1[1], noise_upd1[2],
                                n_sims))
    plt.savefig(figname + '.png', dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()

# 100 units
fig, ax, = plt.subplots(1, 4)
pt_plot = torch.stack(
    [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*4, len_p*5)])
ax[0].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[0].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[0].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[0].set_ylim(ylims)
ax[0].tick_params(axis='x', labelsize=fntsiz-5)
ax[0].tick_params(axis='y', labelsize=fntsiz-5)
ax[0].set_ylabel('Pr of Error', fontsize=fntsiz-3)
ax[0].set_title('{}, grp lr {}'.format(n_units[1], lr_group[0]),
                fontsize=fntsiz-8)
ax[0].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*5, len_p*6)]
ax[1].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[1].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[1].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[1].set_ylim(ylims)
ax[1].tick_params(axis='x', labelsize=fntsiz-5)
ax[1].set_yticklabels([])  # remove ticklables
ax[1].set_xlabel('                Block', fontsize=fntsiz-3)
ax[1].set_title('{}, grp lr {}'.format(n_units[1], lr_group[1]),
                fontsize=fntsiz-8)
ax[1].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*6, len_p*7)]
ax[2].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[2].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[2].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[2].set_ylim(ylims)
ax[2].tick_params(axis='x', labelsize=fntsiz-5)
ax[2].set_yticklabels([])  # remove ticklables
ax[2].set_box_aspect(1)
ax[2].set_title('{}, grp lr {}'.format(n_units[1], lr_group[2]),
                fontsize=fntsiz-8)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*7, len_p*8)]
ax[3].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[3].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[3].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[3].set_ylim(ylims)
ax[3].tick_params(axis='x', labelsize=fntsiz-5)
ax[3].set_yticklabels([])  # remove ticklables
ax[3].set_title('{}, grp lr {}'.format(n_units[1], lr_group[3]),
                fontsize=fntsiz-8)
# ax[3].legend(('{} lesions'.format(n_lesions[0]),
#               '{} lesions'.format(n_lesions[1]),
#               '{} lesions'.format(n_lesions[2])),
#              fontsize=fntsiz-8, bbox_to_anchor=(1.1, 1.), loc="lower left")
ax[3].set_box_aspect(1)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'noise_pt_subplot_k{}_type{}_{}units_noise{}-{}-{}'
                           '_{}sims'.format(
                                sim_prms[1], problem+1, n_units[1],
                                noise_upd1[0], noise_upd1[1], noise_upd1[2],
                                n_sims))
    plt.savefig(figname + '.png', dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()

# 1000 units
fig, ax, = plt.subplots(1, 4)
pt_plot = torch.stack(
    [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*8, len_p*9)])
ax[0].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[0].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[0].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[0].set_ylim(ylims)
ax[0].tick_params(axis='x', labelsize=fntsiz-5)
ax[0].tick_params(axis='y', labelsize=fntsiz-5)
ax[0].set_ylabel('Pr of Error', fontsize=fntsiz-3)
ax[0].set_title('{}, grp lr {}'.format(n_units[2], lr_group[0]),
                fontsize=fntsiz-8)
ax[0].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*9, len_p*10)]
ax[1].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[1].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[1].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[1].set_ylim(ylims)
ax[1].tick_params(axis='x', labelsize=fntsiz-5)
ax[1].set_yticklabels([])  # remove ticklables
ax[1].set_xlabel('                Block', fontsize=fntsiz-3)
ax[1].set_title('{}, grp lr {}'.format(n_units[2], lr_group[1]),
                fontsize=fntsiz-8)
ax[1].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*10, len_p*11)]
ax[2].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[2].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[2].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[2].set_ylim(ylims)
ax[2].tick_params(axis='x', labelsize=fntsiz-5)
ax[2].set_yticklabels([])  # remove ticklables
ax[2].set_box_aspect(1)
ax[2].set_title('{}, grp lr {}'.format(n_units[2], lr_group[2]),
                fontsize=fntsiz-8)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*11, len_p*12)]
ax[3].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[3].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[3].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[3].set_ylim(ylims)
ax[3].tick_params(axis='x', labelsize=fntsiz-5)
ax[3].set_yticklabels([])  # remove ticklables
ax[3].set_title('{}, grp lr {}'.format(n_units[2], lr_group[3]),
                fontsize=fntsiz-8)
# ax[3].legend(('{} lesions'.format(n_lesions[0]),
#               '{} lesions'.format(n_lesions[1]),
#               '{} lesions'.format(n_lesions[2])),
#              fontsize=fntsiz-8, bbox_to_anchor=(1.1, 1.), loc="lower left")
ax[3].set_box_aspect(1)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'noise_pt_subplot_k{}_type{}_{}units_noise{}-{}-{}'
                           '_{}sims'.format(
                                sim_prms[1], problem+1, n_units[2],
                                noise_upd1[0], noise_upd1[1], noise_upd1[2],
                                n_sims))
    plt.savefig(figname + '.png', dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()

# 5000 units
fig, ax, = plt.subplots(1, 4)
pt_plot = torch.stack(
    [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*12, len_p*13)])
ax[0].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[0].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[0].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[0].set_ylim(ylims)
ax[0].tick_params(axis='x', labelsize=fntsiz-5)
ax[0].tick_params(axis='y', labelsize=fntsiz-5)
ax[0].set_ylabel('Pr of Error', fontsize=fntsiz-3)
ax[0].set_title('{}, grp lr {}'.format(n_units[3], lr_group[0]),
                fontsize=fntsiz-8)
ax[0].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*13, len_p*14)]
ax[1].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[1].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[1].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[1].set_ylim(ylims)
ax[1].tick_params(axis='x', labelsize=fntsiz-5)
ax[1].set_yticklabels([])  # remove ticklables
ax[1].set_xlabel('                Block', fontsize=fntsiz-3)
ax[1].set_title('{}, grp lr {}'.format(n_units[3], lr_group[1]),
                fontsize=fntsiz-8)
ax[1].set_box_aspect(1)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*14, len_p*15)]
ax[2].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[2].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[2].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[2].set_ylim(ylims)
ax[2].tick_params(axis='x', labelsize=fntsiz-5)
ax[2].set_yticklabels([])  # remove ticklables
ax[2].set_box_aspect(1)
ax[2].set_title('{}, grp lr {}'.format(n_units[3], lr_group[2]),
                fontsize=fntsiz-8)

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*15, len_p*16)]
ax[3].plot(pt_plot[0], linestyle='-', color=col[problem])
ax[3].plot(pt_plot[1], linestyle=(0, (5, 2.5)), color=col[problem])
ax[3].plot(pt_plot[2], linestyle='dotted', color=col[problem])
ax[3].set_ylim(ylims)
ax[3].tick_params(axis='x', labelsize=fntsiz-5)
ax[3].set_yticklabels([])  # remove ticklables
ax[3].set_title('{}, grp lr {}'.format(n_units[3], lr_group[3]),
                fontsize=fntsiz-8)
# ax[3].legend(('{} lesions'.format(n_lesions[0]),
#               '{} lesions'.format(n_lesions[1]),
#               '{} lesions'.format(n_lesions[2])),
#              fontsize=fntsiz-8, bbox_to_anchor=(1.1, 1.), loc="lower left")
ax[3].set_box_aspect(1)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'noise_pt_subplot_k{}_type{}_{}units_noise{}-{}-{}'
                           '_{}sims'.format(
                                sim_prms[1], problem+1, n_units[3],
                                noise_upd1[0], noise_upd1[1], noise_upd1[2],
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
#      [recr_n[ind_sims[i]].mode() for i in range(len_p*2, len_p*3)],
#      [recr_n[ind_sims[i]].mode() for i in range(len_p*3, len_p*4)]])


# ylims = (recr_avgs.min() - .5, recr_avgs.max() + .5)

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
# # # back to defaults
# # plt.rcdefaults()
