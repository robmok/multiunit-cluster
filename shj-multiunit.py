#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:26:04 2021

Main script to run Shepard et al., 1961's problems, including loading up and
plotting a big simulation 

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

# %% SHJ 6 problems

saveplots = False

saveresults = False

set_seeds = True

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


niter = 10  # 25

# set seeds for niters of shj problem randomised - same seqs across params
if set_seeds:
    seeds = torch.arange(1, niter+1)*10  # - what was used for gridsearch
else:
    seeds = torch.randperm(niter*100)[:niter]


n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
pt_all = torch.zeros([niter, 6, n_epochs])
rec_all =[[] for i in range(6)]
nrec_all = torch.zeros([niter, 6])
w_trace = [[] for i in range(6)]
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

        # model details
        attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
        n_units = 10000 # 3400000 # 2000
        n_dims = inputs.shape[1]
        loss_type = 'cross_entropy'
        k = .005 # .01  # top k percent; .05 = top 5%

        # scale lrs - params determined by n_units=100, k=.01. n_units*k=1
        lr_scale = (n_units * k) / 1

        # 2022
        # final - for plotting
        # tensor([[ 0.2000, 5/11,  3.0000,  0.0750/0.3750,  0.3250,  0.7000]])

        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .2,
            'p': 1,
            'phi': 5.,  # 5/11
            'beta': 1.,
            'lr_attn': 3.,  # .95,  # this scales at grad computation now
            'lr_nn': .375/lr_scale,  # .075/0.3750
            'lr_clusters': .325,
            'lr_clusters_group': .7,
            'k': k
            }
        # OR
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .2,
        #     'p': 1,
        #     'phi': 11.,  # 5/11
        #     'beta': 1.,
        #     'lr_attn': 3.,  # .95,  # this scales at grad computation now
        #     'lr_nn': .075/lr_scale,  # .075/0.3750
        #     'lr_clusters': .325,
        #     'lr_clusters_group': .7,
        #     'k': k
        #     }

        # # fixing attn - scale here now
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .2,
        #     'p': 1,
        #     'phi': 7.,  # 5/11
        #     'beta': 1.,
        #     'lr_nn': .175/lr_scale,  # .075/0.3750
        #     'lr_attn': 3.,
        #     # 'lr_attn': .2,
        #     'lr_clusters': .15,
        #     'lr_clusters_group': .25,
        #     'k': k
        #     }

        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle_seed=seeds[i],
            shj_order=False)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        # - don't save when doing big sim
        w_trace[problem].append(torch.stack(model.fc1_w_trace))
        attn_trace[problem].append(torch.stack(model.attn_trace))

        # save n clusters recruited
        rec_all[problem].append(model.recruit_units_trl)  # saves the seq
        nrec_all[i, problem] = len(model.recruit_units_trl)  # nclus recruited

        print(model.recruit_units_trl)

# for i in range(6):
#     plt.plot(torch.stack(attn_trace[i])[0])
#     plt.show()

# save variables
# - pt_all, nrec_all
if saveresults:
    fn = os.path.join(datadir,
                      'shj_results_pt_nrec_k{}_{}units.pkl'.format(k, n_units))
    
    shj_res = [pt_all, nrec_all]  # seeds_all
    open_file = open(fn, "wb")
    pickle.dump(shj_res, open_file)
    open_file.close()


# the human data from nosofsky, et al. replication
shj = (
    np.array([[0.211, 0.025, 0.003, 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0.],
              [0.378, 0.156, 0.083, 0.056, 0.031, 0.027, 0.028, 0.016,
               0.016, 0.008, 0., 0.002, 0.005, 0.003, 0.002, 0.],
              [0.459, 0.286, 0.223, 0.145, 0.081, 0.078, 0.063, 0.033,
               0.023, 0.016, 0.019, 0.009, 0.008, 0.013, 0.009, 0.013],
              [0.422, 0.295, 0.222, 0.172, 0.148, 0.109, 0.089, 0.062,
               0.025, 0.031, 0.019, 0.025, 0.005, 0., 0., 0.],
              [0.472, 0.331, 0.23, 0.139, 0.106, 0.081, 0.067,
               0.078, 0.048, 0.045, 0.05, 0.036, 0.031, 0.027, 0.016, 0.014],
              [0.498, 0.341, 0.284, 0.245, 0.217, 0.192, 0.192, 0.177,
               0.172, 0.128, 0.139, 0.117, 0.103, 0.098, 0.106, 0.106]])
    )


saveplots = False

fntsiz = 15
ylims = (0., .55)

import matplotlib.font_manager as font_manager
# for roman numerals
font = font_manager.FontProperties(family='Tahoma',
                                   style='normal', size=fntsiz-2)

fig, ax = plt.subplots(2, 1)
ax[0].plot(shj.T)
ax[0].set_ylim(ylims)
ax[0].set_aspect(17)
ax[0].legend(('I', 'II', 'III', 'IV', 'V', 'VI'), fontsize=7)
ax[1].plot(np.nanmean(pt_all, axis=0).T)
ax[1].set_ylim(ylims)
ax[1].set_aspect(17)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'shj_gsearchres_distsq_n94_subplots_{}units_k{}_lr{}'
                           '_grouplr{}_c{}_phi{}_attn{}_nn{}_{}iters.pdf'
                           .format(
                               n_units, k, params['lr_clusters'],
                               params['lr_clusters_group'], params['c'],
                               params['phi'], params['lr_attn'],
                               params['lr_nn'], niter))
    plt.savefig(figname)
plt.show()

# best params by itself
fig, ax = plt.subplots(1, 1)
ax.plot(np.nanmean(pt_all, axis=0).T)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.legend(('I', 'II', 'III', 'IV', 'V', 'VI'), prop=font)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'shj_gsearchres_distsq_{}units_k{}_lr{}_grouplr{}_c{}'
                           '_phi{}_attn{}_nn{}_{}iters.pdf'.format(
                               n_units, k, params['lr_clusters'],
                               params['lr_clusters_group'], params['c'],
                               params['phi'], params['lr_attn'],
                               params['lr_nn'], niter))
    plt.savefig(figname)
plt.show()

# %% load in SHJ results

saveplots = False

figdir = os.path.join(maindir, 'multiunit-cluster_figs')

k = .0005
n_units = 3400000
niter = 25

resdir = os.path.join(maindir, 'muc-shj-gridsearch/')
fn = os.path.join(resdir,
    'shj_results_pt_nrec_k{}_{}units.pkl'.format(k, n_units))

# load [pt_all, nrec_all]
open_file = open(fn, "rb")
shj_res = pickle.load(open_file)
open_file.close()
pt_all = shj_res[0]
nrec_all = shj_res[1]

# plot
fntsiz = 15
ylims = (0., .55)

import matplotlib.font_manager as font_manager
# for roman numerals
font = font_manager.FontProperties(family='Tahoma',
                                   style='normal', size=fntsiz-2)

# best params by itself
fig, ax = plt.subplots(1, 1)
ax.plot(np.nanmean(pt_all, axis=0).T)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.legend(('I', 'II', 'III', 'IV', 'V', 'VI'), prop=font)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                            'shj_gsearch_result_{}units_k{}_{}iters'.format(
                                n_units, k, niter))
    plt.savefig(figname + '.pdf')
    plt.savefig(figname + '.png', dpi=500)
plt.show()

# # n_rec
# plt.boxplot(nrec_all.T)
# plt.show()