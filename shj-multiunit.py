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
from scipy import stats
from scipy import optimize as opt

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

# %%  SHJ single problem

saveplots = False  # 3d plots

plot_seq = 'epoch'  # 'epoch'=plot whole epoch in sections. 'trls'=1st ntrials

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

                # type 1 continuous - 2D
                [[.75,   0,   .75,   0.,  0],
                 [.5,   .25,  .5,   .25,  0],
                 [.25,  .5,   .25,  .5,   1],
                 [0.,   .75,   0.,  .75,  1],
                 [.75,   0.,   0.,  .75,  0],
                 [.5,   .25,  .25,  .5,   0],
                 [.25,  .5,   .5,   .25,  1],
                 [0.,   .75,  .75,  .0,   1]],

                # type 1 continuous - 3D
                [[.75,   0,   .75,   0., .75,  0,  0],
                 [.5,   .25,  .5,   .25, .25, .5,  0],
                 [.25,  .5,   .25,  .5,  .5,  .25, 1],
                 [0.,   .75,   0.,  .75,  0., .75, 1],
                 [.75,   0.,   0.,  .75,  0., .75, 0],
                 [.5,   .25,  .25,  .5,  .5,  .25, 0],
                 [.25,  .5,   .5,   .25, .25, .5,  1],
                 [0.,   .75,  .75,  .0,  .75,  0., 1]],
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
n_units = 500
n_dims = inputs.shape[1]
# nn_sizes = [clus_layer_width, 2]  # only association weights at the end
loss_type = 'cross_entropy'
# c_recruit = 'feedback'  # feedback or loss_thresh

# top k%. so .05 = top 5%
k = .05

# SHJ
# - do I  want to save trace for both clus_pos upadtes? now just saving at the end of both updates

# trials, etc.
n_epochs = 16

# new local attn - scaling lr
lr_scale = (n_units * k) / 1

# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': .9, # w/ attn grad normalized, c can be large now
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 18.5,
#     'beta': 1.,
#     'lr_attn': .15, # this scales at grad computation now
#     'lr_nn': .01/lr_scale,  # scale by n_units*k
#     'lr_clusters': .01,
#     'lr_clusters_group': .1,
#     'k': k
#     }

# # shj params
# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': 1.,  # w/ attn grad normalized, c can be large now
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 12.5,
#     'beta': 1.,
#     'lr_attn': .15,  # this scales at grad computation now
#     'lr_nn': .015/lr_scale,  # scale by n_units*k
#     'lr_clusters': .05,
#     'lr_clusters_group': .1,
#     'k': k
#     }

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
    'lr_clusters_group': .0,
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

# lesioning
lesions = None  # if no lesions
# lesions = {
#     'n_lesions': 10,  # n_lesions per event
#     'gen_rand_lesions_trials': False,  # generate lesion events at random times
#     'pr_lesion_trials': .01,  # if True, set this
#     'lesion_trials': torch.tensor([20])  # if False, set lesion trials
#     }

# noise - mean and sd of noise to be added
# - with update noise, higher lr_group helps save a lot even with few k units. actually didn't add update2 noise though, test again
# - 
noise = None
noise = {'update1': [0, .1],  # unit position updates 1 & 2
          'update2': [0, .0],  # no noise here also makes sense - since there is noise in 1 and you get all that info.
          'recruit': [0., .1],  # recruitment position placement
          'act': [.5, .1]}  # unit activations (non-negative)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=True, lesions=lesions,
    noise=noise)

# # print(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2))
# print(np.unique(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2), axis=0))
# # print(np.unique(np.around(model.attn.detach().numpy()[model.active_units], decimals=2), axis=0))
# print(model.attn)

print(model.recruit_units_trl)
# print(len(model.recruit_units_trl))

# wd='/Users/robert.mok/Documents/Postdoc_cambridge_2020/multiunit-cluster_figs'
# plot for several k values (.01, .05, .1, .2?)
# several n_units (1, 1000, 10000, 1000000) - for n=1, k doesn't matter

# pr target
plt.plot(1 - epoch_ptarget.detach())
plt.ylim([0, .5])
plt.show()

# # attention weights
plt.plot(torch.stack(model.attn_trace, dim=0))
# figname = os.path.join(wd,
#                        'SHJ_attn_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
#                            problem, k, n_units, params['lr_attn'], n_epochs))
# plt.savefig(figname)
plt.show()

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
    n_ims = 20
    plot_trials = torch.tensor(
        torch.linspace(0, len(inputs) * n_epochs, n_ims), dtype=torch.long)
elif plot_seq == 'trls':  # plot first n trials
    plot_n_trials = 50
    plot_trials = torch.arange(plot_n_trials)


# make dir for trial-by-trial images
dn = ('dupd_shj3d_{}_type{}_{}units_k{}_lr{}_grouplr{}_c{}_phi{}_attn{}_nn{}_'
      'upd1noise{}_recnoise{}'.format(
          plot_seq, problem+1, n_units, k, params['lr_clusters'],
          params['lr_clusters_group'], params['c'], params['phi'],
          params['lr_attn'], params['lr_nn'], noise['update1'][1],
          noise['recruit'][1])
      )

if saveplots:
    if not os.path.exists(os.path.join(figdir, dn)):
        os.makedirs(os.path.join(figdir, dn))


# 3d
# https://matplotlib.org/stable/gallery/color/named_colors.html
lims = (0, 1)
# lims = (-.05, 1.05)
for i in plot_trials[0:-1]:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=150)
    ax.scatter(results[i, :, 0],
               results[i, :, 1],
               results[i, :, 2], c='mediumturquoise')  # cornflowerblue / mediumturquoise
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
        figname = os.path.join(figdir, dn, 'trial{}.png'.format(i))
        plt.savefig(figname)

    plt.pause(.25)

# explore lesion units ++ 
# model.units_pos[model.lesion_units[0]] # inspect which units were lesions on lesion trial 0

# %% make gifs

savegif = False

plot_seq = 'epoch'  # epoch/trls

# set params
problem = 4  # 0, 1, 4, 5 right now
lr_clusters = .1
lr_clusters_group = .0
upd1noise = .1  # .1/.2
recnoise = .1  # atm, 0 for dupd, .01 for catlearn

# load from dir
dn = ('dupd_shj3d_{}_type{}_{}units_k{}_lr{}_grouplr{}_c{}_phi{}_attn{}_nn{}_'
      'upd1noise{}_recnoise{}'.format(
          plot_seq, problem+1, n_units, k, lr_clusters,
          lr_clusters_group, params['c'], params['phi'],
          params['lr_attn'], params['lr_nn'], upd1noise,
          recnoise)
      )

if plot_seq == 'epoch':  # plot from start to end in n sections
    n_ims = 20
    plot_trials = torch.tensor(
        torch.linspace(0, len(inputs) * n_epochs, n_ims), dtype=torch.long)
elif plot_seq == 'trls':  # plot first n trials
    plot_n_trials = 50
    plot_trials = torch.arange(plot_n_trials)


images = []
for i in plot_trials[0:-1]:
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


niter = 10
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
    
        # # trying with higher c - flipping 1 & 6
        # # - works well - needs lr_attn to be v slow, then type 6>1 (flipped)
        # # now type II also can be slow, types 3-5 faster - as brad p redicted
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 3.5,  # low = 1; med = 2.2; high = 3.5+
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 1.5,
        #     'beta': 1.,
        #     'lr_attn': .002,  # if too slow, type 1 recruits 4 clus..
        #     'lr_nn': .025/lr_scale,  # scale by n_units*k
        #     'lr_clusters': .01,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # # c param testing new - try to use same phi. adjust lr_nn
        # # low c
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .8,  # w/ attn grad normalized, c can be large now
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 1.5,
        #     'beta': 1.,
        #     'lr_attn': .15,
        #     'lr_nn': .15/lr_scale,  # scale by n_units*k
        #     'lr_clusters': .01,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

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
ax.legend(('1', '2', '3', '4', '5', '6', '1', '2', '3', '4', '5', '6'), fontsize=7)

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
n_units = [20, 100, 1000, 5000]  # [20, 100, 500]
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

# %% plot

saveplots = 0

plt.rcdefaults()

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

# index to average over sims
ind_sims = [torch.arange(i * n_sims, (i + 1) * n_sims)
            for i in range(len(pt) // n_sims)]

# pt
pts = torch.stack(pt)

# average over sims and plot
# - specify sims by how sim_prms are ordered. so 'range' is indexing n_units,
# plotting by n_lesions
len_p = len(n_lesions)

# 20 units
pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(0, len_p)]
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('{} lesions'.format(n_lesions[0]),
                  '{} lesions'.format(n_lesions[1]),
                  '{} lesions'.format(n_lesions[2])))
plt.title('Type {}, {} units'.format(problem + 1, n_units[0]))
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_pt_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[0],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p, len_p*2)]
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('{} lesions'.format(n_lesions[0]),
                  '{} lesions'.format(n_lesions[1]),
                  '{} lesions'.format(n_lesions[2])))
plt.title('Type {}, {} units'.format(problem + 1, n_units[1]))
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_pt_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[1],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()


pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*2, len_p*3)]
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('{} lesions'.format(n_lesions[0]),
                  '{} lesions'.format(n_lesions[1]),
                  '{} lesions'.format(n_lesions[2])))
plt.title('Type {}, {} units'.format(problem + 1, n_units[2]))
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_pt_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[2],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(len_p*3, len_p*4)]
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('{} lesions'.format(n_lesions[0]),
                  '{} lesions'.format(n_lesions[1]),
                  '{} lesions'.format(n_lesions[2])))
plt.title('Type {}, {} units'.format(problem + 1, n_units[3]))
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_pt_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[3],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

# attn
# - are these interpretable if averaged over?
attns = torch.stack(attn_trace)

ylims = (attns.min() - .01, attns.max() + .01)

# 20 units
attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(0, len_p)]
fig, ax = plt.subplots(1, 3)
for iplt in range(len_p):
    ax[iplt].plot(torch.stack(attn_plot)[iplt])
    ax[iplt].set_ylim(ylims)
    ax[iplt].set_title('{} units, {} lesions'.format(n_units[0],
                                                     n_lesions[iplt]),
                       fontsize=10)
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[0],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

# 100 units
attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(len_p, len_p*2)]
fig, ax = plt.subplots(1, 3)
for iplt in range(len_p):
    ax[iplt].plot(torch.stack(attn_plot)[iplt])
    ax[iplt].set_ylim(ylims)
    ax[iplt].set_title('{} units, {} lesions'.format(n_units[1],
                                                     n_lesions[iplt]),
                       fontsize=10)
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[1],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

# 500 units
attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(len_p*2, len_p*3)]
fig, ax = plt.subplots(1, 3)
for iplt in range(len_p):
    ax[iplt].plot(torch.stack(attn_plot)[iplt])
    ax[iplt].set_ylim(ylims)
    ax[iplt].set_title('{} units, {} lesions'.format(n_units[2],
                                                     n_lesions[iplt]),
                       fontsize=10)
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[2],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

attn_plot = [attns[ind_sims[i]].mean(axis=0) for i in range(len_p*3, len_p*4)]
fig, ax = plt.subplots(1, 3)
for iplt in range(len_p):
    ax[iplt].plot(torch.stack(attn_plot)[iplt])
    ax[iplt].set_ylim(ylims)
    ax[iplt].set_title('{} units, {} lesions'.format(n_units[3],
                                                     n_lesions[iplt]),
                       fontsize=10)
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_attn_type{}_trl{}_{}units_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_units[3],
                               n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

# recruit clusters
plt.style.use('seaborn-darkgrid')
recr_n = torch.tensor(
    [len(recruit_trial[i]) for i in range(len(recruit_trial))],  # count
    dtype=torch.float)
ylims = (recr_n.min() - 1, recr_n.max() + 1)

fig, ax, = plt.subplots(2, 2)
recr_plot = [recr_n[ind_sims[i]].mean(axis=0) for i in range(0, len_p)]
ax[0, 0].plot(['0 lesions', '10 lesions', '20 lesions'],
              torch.stack(recr_plot), 'o--')
ax[0, 0].set_title('{} units'.format(n_units[0]))
ax[0, 0].set_ylim(ylims)

recr_plot = [recr_n[ind_sims[i]].mean(axis=0) for i in range(len_p, len_p*2)]
ax[0, 1].plot(['0 lesions', '10 lesions', '20 lesions'],
              torch.stack(recr_plot), 'o--')
ax[0, 1].set_title('{} units'.format(n_units[1]))
ax[0, 1].set_ylim(ylims)

recr_plot = [recr_n[ind_sims[i]].mean(axis=0) for i in range(len_p*2, len_p*3)]
ax[1, 0].plot(['0 lesions', '10 lesions', '20 lesions'],
              torch.stack(recr_plot), 'o--')
ax[1, 0].set_title('{} units'.format(n_units[2]))
ax[1, 0].set_ylim(ylims)

recr_plot = [recr_n[ind_sims[i]].mean(axis=0) for i in range(len_p*3, len_p*4)]
ax[1, 1].plot(['0 lesions', '10 lesions', '20 lesions'],
              torch.stack(recr_plot), 'o--')
ax[1, 1].set_title('{} units'.format(n_units[3]))
ax[1, 1].set_ylim(ylims)
if saveplots:
    figname = os.path.join(figdir,
                           'lesion_recruit_type{}_trl{}_{}sims'.format(
                               problem+1, lesion_trials[0, 0], n_sims))
    plt.savefig(figname, dpi=100)
plt.show()

# back to default
plt.rcdefaults()


# For plotting, make df?

# import pandas as pd
# df_sum = pd.DataFrame(columns=['acc', 'k', 'n_uni'ts, 'n_lesions', 'lesion trials', 'sim_num'])

# %% grid search, fit shj

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

# run all SHJ
beh_seq = shj.T


def negloglik(model_pr, beh_seq):
    return -np.sum(stats.norm.logpdf(beh_seq, loc=model_pr))


# define model to run
# - set up model, run through each shj problem, compute nll
def run_shj_muc(start_params, sim_info, six_problems, beh_seq):
    """
    niter: number of runs per SHJ problem with different sequences (randomised)
    """

    nll_all = torch.zeros(6)
    pt_all = torch.zeros([sim_info['niter'], 6, 16])

    # run niterations, 6 problems
    for i, problem in it.product(range(sim_info['niter']), range(6)):

        stim = six_problems[problem]
        stim = torch.tensor(stim, dtype=torch.float)
        inputs = stim[:, 0:-1]
        output = stim[:, -1].long()  # integer
        # 16 per block
        inputs = inputs.repeat(2, 1)
        output = output.repeat(2).T
        n_dims = inputs.shape[1]

        # initialize model
        model = MultiUnitCluster(sim_info['n_units'], n_dims,
                                 sim_info['attn_type'],
                                 sim_info['k'],
                                 params=None,
                                 fit_params=True, start_params=start_params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, 16, shuffle=True)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

    for problem in range(6):
        nll_all[problem] = negloglik(pt_all[:, problem].mean(axis=0),
                                     beh_seq[:, problem])

    return nll_all.sum(), pt_all.mean(axis=0)  # to run gridsearch and save pt


sim_info = {
    'n_units': 500,
    'attn_type': 'dimensional_local',
    'k': .05,
    'niter': 1  # niter
    }

lr_scale = (n_units * k) / 1

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
ranges = ([torch.arange(1., 1.2, .1),
          torch.arange(1., 1.2, .1),
          torch.arange(.2, .4, .1),
          torch.arange(.0075, .01, .0025) / lr_scale,
          torch.arange(.075, .125, .025),
          torch.arange(.12, .13, .01)])

# set up and save nll, pt, and fit_params
param_sets = torch.tensor(list(it.product(*ranges)))
pt_all = torch.zeros([len(param_sets), 6, 16])
nlls = torch.zeros(len(param_sets))

# grid search
t0 = time.time()
for i, fit_params in enumerate(it.product(*ranges)):
    nlls[i], pt_all[i] = run_shj_muc(
        fit_params, sim_info, six_problems, beh_seq)

t1 = time.time()
print(t1-t0)

# 142.9s - 2 params each, 1 iter. same as above


