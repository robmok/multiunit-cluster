#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:26:04 2021

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools as it
import imageio
import time
# from scipy import stats
# from scipy import optimize as opt
# import pickle

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

# %%  SHJ single problem

saveplots = False  # 3d plots

plot3d = False
plot_seq = 'epoch'  # 'epoch'=plot whole epoch in sections. 'trls'=1st ntrials

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

# set problem
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
n_units = 1000
n_dims = inputs.shape[1]
# nn_sizes = [clus_layer_width, 2]  # only association weights at the end
loss_type = 'cross_entropy'
# c_recruit = 'feedback'  # feedback or loss_thresh

# top k%. so .05 = top 5%
k = .05

# TODO
# - do I  want to save trace for both clus_pos upadtes? now just saving at the
# end of both updates

# trials, etc.
n_epochs = 16

# new local attn - scaling lr
lr_scale = (n_units * k) / 1

# new shj pattern - with phi in the model now
# - editing to show double update effect - mainly lr_group
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .75,
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 9.,
    'beta': 1.,
    'lr_attn': .35,  # this scales at grad computation now
    'lr_nn': .0075/lr_scale,  # scale by n_units*k
    'lr_clusters': .1,
    'lr_clusters_group': .12,
    'k': k
    }

# plotting to compare with nbank model
# low c
# params = {
#     'r': 1,
#     'c': 1.,
#     'p': 1,
#     'phi': 1.5,
#     'beta': 1.,
#     'lr_attn': .35,
#     'lr_nn': .15/lr_scale,
#     'lr_clusters': .05,
#     'lr_clusters_group': .3,
#     'k': k
#     }
# # high c
# params = {
#     'r': 1,
#     'c': 3.,
#     'p': 1,
#     'phi': 1.5,
#     'beta': 1.,
#     'lr_attn': .005,
#     'lr_nn': .025/lr_scale,
#     'lr_clusters': .01,
#     'lr_clusters_group': .1,
#     'k': k
#     }

# new after trying out gridsearch
# tensor([[1.6000, 1.0000, 0.4550, 0.2050, 0.3050, 0.7000]])
# tensor([[0.8000, 1.0000, 0.8000, 0.6500, 0.5000, 0.5000]]) - better

# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': 1.6,  # w/ attn grad normalized, c can be large now
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 1.,
#     'beta': 1.,
#     'lr_attn': .455,  # this scales at grad computation now
#     'lr_nn': .205/lr_scale,  # scale by n_units*k
#     'lr_clusters': .305,  # .075/.1
#     'lr_clusters_group': .7,
#     'k': k
#     }

# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': .8,  # w/ attn grad normalized, c can be large now
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 1.,
#     'beta': 1.,
#     'lr_attn': .8,  # this scales at grad computation now
#     'lr_nn': .65/lr_scale,  # scale by n_units*k
#     'lr_clusters': .5,  # .075/.1
#     'lr_clusters_group': .5,
#     'k': k
#     }
# # tensor([[0.8000, 1.0000, 0.8500, 0.6500, 0.4500, 0.9000]])
# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': .8,  # w/ attn grad normalized, c can be large now
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 1.,
#     'beta': 1.,
#     'lr_attn': .85,  # this scales at grad computation now
#     'lr_nn': .65/lr_scale,  # scale by n_units*k
#     # 'lr_nn': .15*k,  # scale by k
#     'lr_clusters': .45,  # .075/.1
#     'lr_clusters_group': .9,
#     'k': k
#     }

# # testing for noise plots
# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': .75,
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 9.,
#     'beta': 1.,
#     'lr_attn': .35,  # this scales at grad computation now
#     'lr_nn': .0075/lr_scale,  # scale by n_units*k
#     'lr_clusters': .2,
#     'lr_clusters_group': .4,
#     'k': k
#     }

# best params, slower lr_clus
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .8,  # w/ attn grad normalized, c can be large now
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 1.,
    'beta': 1.,
    'lr_attn': .8,  # this scales at grad computation now
    'lr_nn': .65/lr_scale,  # scale by n_units*k
    'lr_clusters': .15,  # .075/.1
    'lr_clusters_group': .5,
    'k': k
    }

# # tensor([ 2.0000, 17.0000,  0.9500,  0.9500,  0.9500,  0.9000])
# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': 2.,  # w/ attn grad normalized, c can be large now
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 17.,
#     'beta': 1.,
#     'lr_attn': .95,  # this scales at grad computation now
#     'lr_nn': .95/lr_scale,  # scale by n_units*k
#     'lr_clusters': .95,  # .075/.1
#     'lr_clusters_group': .9,
#     'k': k
#     }

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .4,
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 7.,
    'beta': 1.,
    'lr_attn': .95,  # this scales at grad computation now
    'lr_nn': .05/lr_scale,  # scale by n_units*k
    # 'lr_nn': .15*k,  # scale by k
    'lr_clusters': .35,  # .075/.1
    'lr_clusters_group': .9,
    'k': k
    }

# editing one of gsearch results, looks ok
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .3,
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 3.5,
    'beta': 1.,
    'lr_attn': .95,  # this scales at grad computation now
    'lr_nn': .25/lr_scale,  # scale by n_units*k
    'lr_clusters': .35,  # .075/.1
    'lr_clusters_group': .9,
    'k': k
            }
# lesioning
lesions = None  # if no lesions
# lesions = {
#     'n_lesions': 10,  # n_lesions per event
#     'gen_rand_lesions_trials': False,  # generate lesion events at random times
#     'pr_lesion_trials': .01,  # if True, set this
#     'lesion_trials': torch.tensor([20])  # if False, set lesion trials
#     }

# noise - mean and sd of noise to be added
# - with update noise, higher lr_group helps save a lot even with few k units.
# actually didn't add update2 noise though, test again
noise = None
noise = {'update1': [0, .15],  # . 1unit position updates 1 & 2
          'update2': [0, .0],  # no noise here also makes sense - since there is noise in 1 and you get all that info.
          'recruit': [0., .1],  # .1 recruitment position placement
          'act': [.5, .1]}  # unit activations (non-negative)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, lesions=lesions,
    noise=noise, shj_order=True)


# # print(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2))
# print(np.unique(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2), axis=0))
# # print(np.unique(np.around(model.attn.detach().numpy()[model.active_units], decimals=2), axis=0))
# print(model.attn)

print(model.recruit_units_trl)
# print(len(model.recruit_units_trl))
print(epoch_ptarget)

# pr target
plt.plot(1 - epoch_ptarget.detach())
plt.ylim([0, .5])
plt.show()

# # attention weights
# plt.plot(torch.stack(model.attn_trace, dim=0))
# # figname = os.path.join(figdir,
# #                        'SHJ_attn_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
# #                            problem, k, n_units, params['lr_attn'], n_epochs))
# # plt.savefig(figname)
# plt.show()

# # unit positions
# results = torch.stack(model.units_pos_trace, dim=0)[-1, model.active_units]
# plt.scatter(results[:, 0], results[:, 1])
# # plt.xlim([-1, 1])
# # plt.ylim([-1, 1])
# plt.gca().set_aspect('equal', adjustable='box')
# # plt.axis('equal')
# plt.show()

# plot 3d - unit positions over time
results = torch.stack(
    model.units_pos_bothupd_trace, dim=0)[:, model.active_units]

if plot_seq == 'epoch':  # plot from start to end in n sections
    n_ims = 9 # 9 = 1 im per 2 blocks (16 trials * 2 (2nd update))
    plot_trials = torch.tensor(
        torch.linspace(0, len(inputs) * n_epochs, n_ims), dtype=torch.long)

    # problem=2/3, 6 clus needed this
    # n_ims = 18 # full
    # plot_trials = torch.tensor(
    #     torch.linspace(0, len(inputs) * n_epochs * 2, n_ims), dtype=torch.long)
    # plot_trials[-1] = plot_trials[-1]-1  # last trial

elif plot_seq == 'trls':  # plot first n trials
    plot_n_trials = 80
    plot_trials = torch.arange(plot_n_trials)

# 3d
# make dir for trial-by-trial images
if noise and saveplots:
    dn = ('dupd_shj3d_{}_type{}_{}units_k{}_lr{}_grouplr{}_c{}_phi{}_attn{}_'
          'nn{}_upd1noise{}_recnoise{}'.format(
              plot_seq, problem+1, n_units, k, params['lr_clusters'],
              params['lr_clusters_group'], params['c'], params['phi'],
              params['lr_attn'], params['lr_nn'], noise['update1'][1],
              noise['recruit'][1])
          )

    if not os.path.exists(os.path.join(figdir, dn)):
        os.makedirs(os.path.join(figdir, dn))

if plot3d:
    lims = (0, 1)
    # lims = (-.05, 1.05)
    for i in plot_trials:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=150)
        ax.scatter(results[i, :, 0],
                   results[i, :, 1],
                   results[i, :, 2], c=col[problem])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_zlim(lims)

        # keep grid lines, remove labels
        # # labels = ['', '', '', '', '', '']
        labels = [0, '', '', '', '', 1]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_zticklabels(labels)

        # remove grey color - white
        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        # save
        if saveplots:
            figname = os.path.join(figdir, dn, 'trial{}'.format(i))
            plt.savefig(figname + '.png')
            plt.savefig(figname + '.pdf')

        plt.pause(.2)

# explore lesion units ++
# model.units_pos[model.lesion_units[0]] # inspect which units were lesions on
# lesion trial 0

# %% make gifs

savegif = True

plot_seq = 'trls'  # epoch/trls

# set params
problem = 5
lr_clusters = .15
lr_clusters_group = .5
upd1noise = .15  # .1/.15/.2
recnoise = .1

# load from dir
dn = ('dupd_shj3d_{}_type{}_{}units_k{}_lr{}_grouplr{}_c{}_phi{}_attn{}_nn{}_'
      'upd1noise{}_recnoise{}'.format(
          plot_seq, problem+1, n_units, k, lr_clusters,
          lr_clusters_group, params['c'], params['phi'],
          params['lr_attn'], params['lr_nn'], upd1noise,
          recnoise)
      )

if plot_seq == 'epoch':  # plot from start to end in n sections
    n_ims = 9  # 9 / 20
    plot_trials = torch.tensor(
        torch.linspace(0, len(inputs) * n_epochs, n_ims), dtype=torch.long)

    # # problem 2/3, 6 clus needed this
    # n_ims = 18  # full
    # plot_trials = torch.tensor(
    #     torch.linspace(0, len(inputs) * n_epochs * 2, n_ims),
    #     dtype=torch.long)
    # plot_trials[-1] = plot_trials[-1]-1  # last trial

elif plot_seq == 'trls':  # plot first n trials
    plot_n_trials = 80
    plot_trials = torch.arange(plot_n_trials)


images = []
for i in plot_trials:
    fname = os.path.join(figdir, dn, 'trial{}.png'.format(i))
    images.append(imageio.imread(fname))

if savegif:
    imageio.mimsave(
        os.path.join(figdir, dn, 'trials.gif'), images, duration=.4)

# %% SHJ 6 problems

saveplots = False

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


niter = 1
n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
pt_all = torch.zeros([niter, 6, n_epochs])
w_trace = [[] for i in range(6)]
attn_trace = [[] for i in range(6)]

# t0 = time.time()

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
        n_units = 1000
        n_dims = inputs.shape[1]
        loss_type = 'cross_entropy'
        k = .05  # top k%. so .05 = top 5%

        # scale lrs - params determined by n_units=100, k=.01. n_units*k=1
        lr_scale = (n_units * k) / 1

        # new shj pattern - with phi in the model now
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .7,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 9.,
            'beta': 1.,
            'lr_attn': .35,  # this scales at grad computation now
            'lr_nn': .0075/lr_scale,  # scale by n_units*k
            'lr_clusters': .075,  # .075/.1
            'lr_clusters_group': .12,
            'k': k
            }

        # # high c
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 3.,
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 1.5,
        #     'beta': 1.,
        #     'lr_attn': .002,
        #     'lr_nn': .025/lr_scale,  # scale by n_units*k
        #     'lr_clusters': .01,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # # comparing with n_banks model
        # # low c
        # params = {
        #     'r': 1,
        #     'c': .75,
        #     'p': 1,
        #     'phi': 1.3,
        #     'beta': 1,
        #     'lr_attn': .2,
        #     'lr_nn': .1/lr_scale,
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # # high c
        # params = {
        #     'r': 1,
        #     'c': 2.6,
        #     'p': 1,
        #     'phi': 1.1,
        #     'beta': 1,
        #     'lr_attn': .002,
        #     'lr_nn': .02/lr_scale,  # .01/.02
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # # v2
        # # low c
        # params = {
        #     'r': 1,
        #     'c': .75,
        #     'p': 1,
        #     'phi': 1.,
        #     'beta': 1,
        #     'lr_attn': .2,
        #     'lr_nn': .1/lr_scale,
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # # high c
        # params = {
        #     'r': 1,
        #     'c': 2.5,
        #     'p': 1,
        #     'phi': 2.,
        #     'beta': 1,
        #     'lr_attn': .005,
        #     'lr_nn': .002/lr_scale,
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # new after trying out gridsearch

        # tensor([[1.8000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]]) # mse
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': 1.8,
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1.,
            'beta': 1.,
            'lr_attn': .55,  # this scales at grad computation now
            'lr_nn': .15/lr_scale,  # scale by n_units*k
            # 'lr_nn': .15*k,  # scale by k
            'lr_clusters': .15,  # .075/.1
            'lr_clusters_group': .9,
            'k': k
            }
        # tensor([[0.4000, 7.0000, 0.9500, 0.0500, 0.3500, 0.9000]])# sse- seems more stable with recruited clusters
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .4,
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 7.,
            'beta': 1.,
            'lr_attn': .95,  # this scales at grad computation now
            'lr_nn': .05/lr_scale,  # scale by n_units*k
            # 'lr_nn': .15*k,  # scale by k
            'lr_clusters': .35,  # .075/.1
            'lr_clusters_group': .9,
            'k': k
            }

        # new gridsearch
        # tensor([[0.4000, 3.0000, 0.4500, 0.2500, 0.3500, 0.9000]])
        # tensor([[0.4000, 7.0000, 0.5500, 0.0500, 0.4500, 0.9000]])
        
        # tensor([[0.4000, 3.0000, 0.6500, 0.2500, 0.4500, 0.9000]])
        # tensor([[0.4000, 7.0000, 0.9500, 0.0500, 0.3500, 0.9000]])
        # tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]])

        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .4,
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 3.,
            'beta': 1.,
            'lr_attn': .45,  # this scales at grad computation now
            'lr_nn': .25/lr_scale,  # scale by n_units*k
            'lr_clusters': .35,  # .075/.1
            'lr_clusters_group': .9,
            'k': k
            }

        # editing above to make better
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .3,
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 3.5,
            'beta': 1.,
            'lr_attn': .95,  # this scales at grad computation now
            'lr_nn': .25/lr_scale,  # scale by n_units*k
            'lr_clusters': .35,  # .075/.1
            'lr_clusters_group': .9,
            'k': k
            }

        # tensor([[0.3000, 0.7500, 0.9500, 0.3500, 0.4500, 0.9000]])  # finegsearrch dist - same as before type 3 fast
        # tensor([[0.3000, 0.7500, 0.9500, 0.2500, 0.4500, 0.9000]])  # finegsearch dist, slightly slower
        
        # tensor([[0.7000, 1.0000, 0.1500, 0.6500, 0.7500, 0.1000/0.3000]])  # gsearch dist**2 - slower, gd pattern
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .3,
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 2.5,
            'beta': 1.,
            'lr_attn': .95,  # this scales at grad computation now
            'lr_nn': .35/lr_scale,  # scale by n_units*k
            'lr_clusters': .45,
            'lr_clusters_group': .9,
            'k': k
            }

        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs,  # shuffle_seed=2,
            shj_order=True)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        w_trace[problem].append(torch.stack(model.fc1_w_trace))
        attn_trace[problem].append(torch.stack(model.attn_trace))

        print(model.recruit_units_trl)

# t1 = time.time()
# print(t1-t0)

plt.plot(np.nanmean(pt_all, axis=0).T)
plt.ylim([0., 0.55])
plt.gca().legend(('1', '2', '3', '4', '5', '6'))
plt.show()

# for i in range(6):
#     plt.plot(torch.stack(attn_trace[i])[0])
#     plt.show()

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

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(shj.T)
# ax[0].set_ylim([0., .55])
# ax[0].set_aspect(17)
# ax[1].plot(pt_all.mean(axis=0).T)
# ax[1].set_ylim([0., .55])
# ax[1].legend(('1', '2', '3', '4', '5', '6'), fontsize=7)
# ax[1].set_aspect(17)
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.plot(shj.T, 'k')
# ax.plot(pt_all.mean(axis=0).T, 'o-')
# # ax.plot(pt_all[0:10].mean(axis=0).T, 'o-')
# ax.set_ylim([0., .55])
# ax.legend(('1', '2', '3', '4', '5', '6', '1', '2', '3', '4', '5', '6'),
#           fontsize=7)

# %% plotting weights to compare to nbank model

# i = 0
# problem = 5

# ylims = (-torch.max(torch.abs(w)), torch.max(torch.abs(w)))
ylims = (-.06, .06)

for problem in range(6):
    w = w_trace[problem][i]
    w0 = torch.reshape(w, (w.shape[0], w.shape[1] * w.shape[2]))
    plt.plot(w0[:, torch.nonzero(w0.sum(axis=0)).squeeze()])
    plt.ylim(ylims)
    plt.title('assoc ws, type {}, c = {}'.format(problem+1, params['c']))
    if saveplots:
        figname = (
            os.path.join(figdir,
                         'shj_assocw_type{}_c{}_k{}_{}units.pdf'.format(
                             problem+1, params['c'], k, n_units))
        )
        plt.savefig(figname)
    plt.show()

