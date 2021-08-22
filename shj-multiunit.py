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
# import time
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
n_units = 2000
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
    'lr_clusters_group': .4,
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
noise = {'update1': [0, .2],  # . 1unit position updates 1 & 2
          'update2': [0, .0],  # no noise here also makes sense - since there is noise in 1 and you get all that info.
          'recruit': [0., .1],  # .1 recruitment position placement
          'act': [.5, .1]}  # unit activations (non-negative)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=False, lesions=lesions,
    noise=noise, shj_order=True)

# # print(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2))
# print(np.unique(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2), axis=0))
# # print(np.unique(np.around(model.attn.detach().numpy()[model.active_units], decimals=2), axis=0))
# print(model.attn)

print(model.recruit_units_trl)
# print(len(model.recruit_units_trl))


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

savegif = False

plot_seq = 'trls'  # epoch/trls

# set params
problem = 0
lr_clusters = .1
lr_clusters_group = .4
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


niter = 20
n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
pt_all = torch.zeros([niter, 6, n_epochs])
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
        n_units = 500
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
        # tensor([[1.6000, 1.0000, 0.4550, 0.2050, 0.3050, 0.7000]])
        # tensor([[0.8000, 1.0000, 0.8000, 0.6500, 0.5000, 0.5000]]) - better

        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': 1.6,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1.,
            'beta': 1.,
            'lr_attn': .455,  # this scales at grad computation now
            'lr_nn': .205/lr_scale,  # scale by n_units*k
            'lr_clusters': .305,  # .075/.1
            'lr_clusters_group': .7,
            'k': k
            }

        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .8,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1.,
            'beta': 1.,
            'lr_attn': .8,  # this scales at grad computation now
            'lr_nn': .65/lr_scale,  # scale by n_units*k
            'lr_clusters': .5,  # .075/.1
            'lr_clusters_group': .5,
            'k': k
            }

        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle=True)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        w_trace[problem].append(torch.stack(model.fc1_w_trace))
        attn_trace[problem].append(torch.stack(model.attn_trace))

        print(model.recruit_units_trl)

plt.plot(pt_all.mean(axis=0).T)
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

fig, ax = plt.subplots(1, 1)
ax.plot(shj.T, 'k')
ax.plot(pt_all.mean(axis=0).T, 'o-')
# ax.plot(pt_all[0:10].mean(axis=0).T, 'o-')
ax.set_ylim([0., .55])
ax.legend(('1', '2', '3', '4', '5', '6', '1', '2', '3', '4', '5', '6'),
          fontsize=7)

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


# %% lesioning experiments

saveplots = 0

problem = 1

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

lesions = {
    'n_lesions': 10,  # n_lesions per event
    'gen_rand_lesions_trials': False,  # generate lesion events at random times
    'pr_lesion_trials': .01,  # if True, set this
    'lesion_trials': torch.tensor([20])  # if False, set lesion trials
    }

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

n_sims = 20
shuffle_seeds = torch.randperm(n_sims*5)[:n_sims]

# things to manipulate
#  - with 5000/8000 recovers - actually even better (recruit extra cluster so
# higher act... feature/bug? could be feature: learning, hpc synpase overturn)
n_units = [20, 100, 1000, 10000]  # [20, 100, 500]
k = [.05]
n_lesions = [0, 25, 50]
lesion_trials = np.array([[60]])  # [60]]  # 1 per lesion, but do at diff times

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
        #     'lr_clusters_group': .5,
        #     'k': sim_prms[1]
        #     }

        model = MultiUnitCluster(sim_prms[0], n_dims, attn_type, sim_prms[1],
                                 params=params)

        lesions = {
            'n_lesions': sim_prms[3],  # n_lesions per event
            'gen_rand_lesions_trials': False,  # lesion events at random times
            'pr_lesion_trials': .01,  # if True, set this
            'lesion_trials': torch.tensor(sim_prms[2])  # if False, set this
            }

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle=True,
            shuffle_seed=shuffle_seeds[isim], lesions=lesions)

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

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p, len_p*2)]
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
                           'units_{}sims'.format(
                               sim_prms[1], problem+1, lesion_trials[0, 0],
                               n_units[0],  n_units[1], n_units[2], n_units[3],
                               n_sims))
    plt.savefig(figname + '.png', dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()

# recruit clusters
# plt.style.use('seaborn-darkgrid')
recr_n = torch.tensor(
    [len(recruit_trial[i]) for i in range(len(recruit_trial))],  # count
    dtype=torch.float)

recr_avgs = torch.tensor(
    [[recr_n[ind_sims[i]].mode() for i in range(0, len_p)],
     [recr_n[ind_sims[i]].mode() for i in range(len_p*2, len_p*3)],
     [recr_n[ind_sims[i]].mode() for i in range(len_p*3, len_p*4)]])


ylims = (recr_avgs.min() - .5, recr_avgs.max() + .5)

mrksiz = 4

fig, ax, = plt.subplots(1, 4)
recr_plot = torch.stack(
    [recr_n[ind_sims[i]].mode().values for i in range(0, len_p)])
ax[0].plot(['0', '10', '20'], recr_plot, 'o--', color=col[problem],
           markersize=mrksiz)
ax[0].set_title('{} units'.format(n_units[0]), fontsize=fntsiz-5)
ax[0].tick_params(axis='x', labelsize=fntsiz-5)
ax[0].tick_params(axis='y', labelsize=fntsiz-5)
ax[0].set_ylabel('No. of recruitments', fontsize=fntsiz-3)
ax[0].set_ylim(ylims)
ax[0].set_box_aspect(1)

recr_plot = torch.stack(
    [recr_n[ind_sims[i]].mode().values for i in range(len_p, len_p*2)])
ax[1].plot(['0', '10', '20'], recr_plot, 'o--', color=col[problem],
           markersize=mrksiz)
ax[1].set_title('{} units'.format(n_units[1]), fontsize=fntsiz-5)
ax[1].tick_params(axis='x', labelsize=fntsiz-5)
ax[1].set_yticklabels([])  # remove ticklables
ax[1].set_ylim(ylims)
ax[1].set_xlabel('                    No. of lesions', fontsize=fntsiz-3)
ax[1].set_box_aspect(1)

recr_plot = torch.stack(
    [recr_n[ind_sims[i]].mode().values for i in range(len_p*2, len_p*3)])
ax[2].plot(['0', '10', '20'], recr_plot, 'o--',
           color=col[problem], markersize=mrksiz)
ax[2].set_title('{} units'.format(n_units[2]), fontsize=fntsiz-5)
ax[2].tick_params(axis='x', labelsize=fntsiz-5)
ax[2].set_yticklabels([])  # remove ticklables
ax[2].set_ylim(ylims)
ax[2].set_box_aspect(1)

recr_plot = torch.stack(
    [recr_n[ind_sims[i]].mode().values for i in range(len_p*3, len_p*4)])
ax[3].plot(['0', '10', '20'], recr_plot, 'o--', color=col[problem],
           markersize=mrksiz)
ax[3].set_title('{} units'.format(n_units[3]), fontsize=fntsiz-5)
ax[3].tick_params(axis='x', labelsize=fntsiz-5)
ax[3].set_yticklabels([])  # remove ticklables
ax[3].set_ylim(ylims)
ax[3].set_box_aspect(1)
plt.tight_layout()

if saveplots:
    figname = os.path.join(figdir,
                           'lesion_recruit_k{}_type{}_trl{}_{}-{}-{}-{}units_'
                           '{}sims'.format(
                               sim_prms[1], problem+1, lesion_trials[0, 0],
                               n_units[0], n_units[1], n_units[2], n_units[3],
                               n_sims))
    plt.savefig(figname + '.png', dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()
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
