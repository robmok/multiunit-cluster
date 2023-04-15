#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:28:33 2023

Script to plot single problems including lesion and noise plots (Fig. S3, S4)

And gifs of units learning positions - Fig. S2

@author: robert.mok
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import imageio

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

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
n_units = 50000
n_dims = inputs.shape[1]
# nn_sizes = [clus_layer_width, 2]  # only association weights at the end
loss_type = 'cross_entropy'
# c_recruit = 'feedback'  # feedback or loss_thresh

# top k%. so .05 = top 5%
k = .005  # .05

# trials, etc.
n_epochs = 16

# new local attn - scaling lr
lr_scale = (n_units * k) / 1

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .2,
    'p': 1,
    'phi': 7.,  # 5/11
    'beta': 1.,
    'lr_nn': .175/lr_scale,  # .075/0.3750
    # 'lr_attn': .5/(n_units*k), # 3., # maybe should scale here..!
    'lr_attn': .4,
    'lr_clusters': .05,
    'lr_clusters_group': .25,
    'k': k
    }

# lesioning
lesions = None  # None if no lesions, True otherwise
if lesions:
    lesions = {
        'n_lesions': 10,  # n_lesions per event
        'gen_rand_lesions_trials': False,  # generate lesion events at random times
        'pr_lesion_trials': .01,  # if True, set this
        'lesion_trials': torch.tensor([20])  # if False, set lesion trials
        }

# noise - mean and sd of noise to be added
# - with update noise, higher lr_group helps save a lot even with few k units.
# actually didn't add update2 noise though, test again
noise = None  # None if no noise, True otherwise
if noise:
    noise = {'update1': [0, .15],  # . 1unit position updates 1 & 2
              'update2': [0, .0],  # no noise here also makes sense - since there is noise in 1 and you get all that info.
              'recruit': [0., .1],  # .1 recruitment position placement
              'act': [.5, .1]}  # unit activations (non-negative)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle_seed=1,
    lesions=lesions,
    noise=noise, shj_order=False)

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

# attention weights
plt.plot(torch.stack(model.attn_trace, dim=0))
# figname = os.path.join(figdir,
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
    for i in plot_trials:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=150)
        ax.scatter(results[i, :, 0],
                   results[i, :, 1],
                   results[i, :, 2], c=col[problem])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_zlim(lims)

        # keep grid lines, remove labels
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

# %% make gifs

savegif = False

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