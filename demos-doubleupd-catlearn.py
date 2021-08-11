#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:25:48 2021

Toy examples showing double-update

- in presence of update noise (update 1 & 2), recruitment placement noise

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import imageio

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

# %% double update demo

saveplots = False

# one 2D gaussian - k units update

# # sample from a 2D gaussian
# mu1 = np.array([.25, .75])
# var1 = np.array([.005, .005])
# cov1 = 0

# larger
mu1 = np.array([0.5, 0.5])
var1 = np.array([.0025, .0025])
cov1 = 0

x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
rv1 = multivariate_normal([mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]])

# % sampling
npoints = 50
x1 = np.random.multivariate_normal(
    [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax1.contour(x, y, rv1.pdf(pos), cmap='Blues')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_facecolor((.8, .8, .8))
ax1.scatter(x1[:, 0], x1[:, 1],
            c=np.expand_dims(np.array([.4, .4, .4]), axis=0),
            marker='x', s=7, linewidth=.75)
ax1.set_aspect('equal', adjustable='box')

# assign label - just one 'category'
inputs = torch.tensor(x1, dtype=torch.float32)
output = torch.zeros(npoints, dtype=torch.long)

n_epochs = 1

# double update demo

# model
attn_type = 'dimensional_local'
n_units = 100
n_dims = inputs.shape[1]
loss_type = 'cross_entropy'
k = .05

# scaling lr
lr_scale = (n_units * k) / 1

params = {
    'r': 1,
    'c': .5,
    'p': 1,
    'phi': 2.5,
    'beta': 1.,
    'lr_attn': .0,
    'lr_nn': .1/lr_scale,
    'lr_clusters': .1,
    'lr_clusters_group': .5,
    'k': k
    }

lesions = None

# noise - mean and sd of noise to be added
noise = None
noise = {'update1': [0, .09],  # unit position updates 1 & 2
         'update2': [0, .0],  # no noise here also makes sense
         'recruit': [0., .035],  # recruitment position placement
         'act': [.0, .0]}  # unit activations (non-negative)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=False, lesions=lesions,
    noise=noise)

"""
- looking good
- annealing / make the lr's small enough - works... unless lr_group too big
since that makes it stick around where most dots are.

- compare with lr_group vs no lr_group effects: with upd1 noise sd=1, can see
this. lr_clusters = .05, lr_group = .2-.5, good. no lr_group means units
keep going around, whereas lr_group model settles to the centre pretty quickly

- more noise, e.g. sd=2., also works, but now every trial is just expanding
due to noise, then going together. Shows it works to combat noise, but not
sure the brain does this...? TBF, the no lr_group does a bad job with this.
needs lr_group higher - .35-5

"""

# plot both updates

# have each trial twice - to plot with double update with same stim
inputs_d = torch.stack([np.repeat(inputs[:, 0], 2),
                        np.repeat(inputs[:, 1], 2)]).T

results = torch.stack(
    model.units_pos_bothupd_trace, dim=0)[:, model.active_units]

# plot_trials = torch.tensor(torch.linspace(0, len(inputs) * n_epochs, 10),
#                             dtype=torch.long)

plot_trials = torch.arange(20)

# make dir for trial-by-trial images
dn = 'demos_dupd_{}units_k{}_lr{}_grouplr{}_upd1noise{}_recnoise{}'.format(
     n_units, k, params['lr_clusters'], params['lr_clusters_group'],
     noise['update1'][1], noise['recruit'][1])
if saveplots:
    if not os.path.exists(os.path.join(figdir, dn)):
        os.makedirs(os.path.join(figdir, dn))

for i in plot_trials:

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    # plot distribution stimuli come from (2d gaussian)
    # ax.contour(x, y, rv1.pdf(pos), cmap='Greys', alpha=.1)

    # stimulus pos on trial i
    if np.mod(i, 2) == 0:
        m = '$S$'  # stim
        c = 'black'
        ax.scatter(inputs_d[i, 0], inputs_d[i, 1],
                   c=c, marker=m, s=1000, linewidth=.05, zorder=3)

    else:
        m = '$c$'  # second update
        c = 'black' # np.array([[.5, .5, .5],])
        ax.scatter(results[i, :, 0].mean(), results[i, :, 1].mean(),c=c,
                   marker=m, edgecolor='black', s=1000, linewidth=.05, zorder=3)


    # unit pos on trial i - showing double update
    ax.scatter(results[i, :, 0], results[i, :, 1],
               s=1000, edgecolors='black', linewidth=.5, zorder=2, alpha=.75)

    # ax.set_facecolor((.8, .8, .8))
    ax.set_xlim([.35, .65])
    ax.set_ylim([.35, .65])
    # labels = [0, '', '', '', '', 1]
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.axis('off')
    ax.set_aspect('equal', adjustable='box')

    # save
    if saveplots:
        figname = os.path.join(figdir, dn, 'trial{}.png'.format(i))
        plt.savefig(figname)

    plt.pause(.25)

# %% concept learning toy example

saveplots = True

# make 2 categories - sample from two 2D gaussians
mu1 = np.array([.75, .75])
var1 = np.array([.005, .005])
cov1 = 0
mu2 = np.array([.25, .6])
var2 = np.array([.005, .005])
cov2 = 0

x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
rv1 = multivariate_normal([mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]])
rv2 = multivariate_normal([mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]])

# sampling
npoints = 50
x1 = np.random.multivariate_normal(
    [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)
x2 = np.random.multivariate_normal(
    [mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]], npoints)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax1.contour(x, y, rv1.pdf(pos), cmap='Blues')
ax1.contour(x, y, rv2.pdf(pos), cmap='Oranges')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_facecolor((.5, .5, .5))
ax1.scatter(x1[:, 0], x1[:, 1], marker='x', c='blue', s=7, linewidth=.75)
ax1.scatter(x2[:, 0], x2[:, 1], marker='x', c='green', s=7, linewidth=.75)
ax1.set_aspect('equal', adjustable='box')
plt.show()

# 3d plot
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(x, y, rv1.pdf(pos), cmap='GnBu', alpha=.75)
# ax.plot_surface(x, y, rv2.pdf(pos), cmap='RdPu', alpha=.5)
# ax.grid(False)
# ax.set_zticks([])
# plt.show()

# assign category label (output)
inputs = torch.cat([torch.tensor(x1, dtype=torch.float32),
                    torch.tensor(x2, dtype=torch.float32)])
output = torch.cat([torch.zeros(npoints, dtype=torch.long),
                    torch.ones(npoints, dtype=torch.long)])

n_epochs = 1

# TODO
# do this in train later - just need to save the shuffled inputs/outputs
shuffle_ind = torch.randperm(len(inputs))
inputs = inputs[shuffle_ind]
output = output[shuffle_ind]

# model
attn_type = 'dimensional_local'
n_units = 500
n_dims = inputs.shape[1]
loss_type = 'cross_entropy'
k = .05

lr_scale = (n_units * k) / 1

params = {
    'r': 1,
    'c': .25,
    'p': 1,
    'phi': 2.5,
    'beta': 1.,
    'lr_attn': .0,
    'lr_nn': .1/lr_scale,
    'lr_clusters': .1,
    'lr_clusters_group': .4,
    'k': k
    }

lesions = None

# noise - mean and sd of noise to be added
noise = None
noise = {'update1': [0, .1],  # unit position updates 1 & 2
         'update2': [0, .0],  # no noise here also makes sense
         'recruit': [0., .01],  # recruitment position placement
         'act': [.0, .0]}  # unit activations (non-negative)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=False, lesions=lesions,
    noise=noise)

# pr target
plt.plot(1 - trial_ptarget.detach())
plt.ylim([0, .5])
plt.show()

# # attention weights
# plt.plot(torch.stack(model.attn_trace, dim=0))
# plt.show()

# plot both updates

# each trial presented twice - to plot with double update
inputs_d = torch.stack([np.repeat(inputs[:, 0], 2),
                        np.repeat(inputs[:, 1], 2)]).T

results = torch.stack(
    model.units_pos_bothupd_trace, dim=0)[:, model.active_units]

# plot_trials = torch.tensor(torch.linspace(0, len(inputs) * n_epochs, 10),
#                             dtype=torch.long)

plot_trials = torch.arange(50)

# make dir for trial-by-trial images
dn = 'demos_catlearn_{}units_k{}_lr{}_grouplr{}_upd1noise{}_recnoise{}'.format(
     n_units, k, params['lr_clusters'], params['lr_clusters_group'],
     noise['update1'][1], noise['recruit'][1])
if saveplots:
    if not os.path.exists(os.path.join(figdir, dn)):
        os.makedirs(os.path.join(figdir, dn))

for i in plot_trials:

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    # plot distribution stimuli come from (2d gaussian)
    ax.contour(x, y, rv1.pdf(pos), cmap='Blues', alpha=.75)
    ax.contour(x, y, rv2.pdf(pos), cmap='Oranges', alpha=.75)

    # stimulus pos on trial i
    ax.scatter(inputs_d[i, 0],
               inputs_d[i, 1],
               c='black', marker='x', s=25, linewidth=1.2, zorder=3)

    # unit pos on trial i - showing double update
    ax.scatter(results[i, :, 0], results[i, :, 1],
               c='grey', edgecolors='black', linewidth=.2, s=8, zorder=2)

    # ax.set_facecolor((.85, .85, .85))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')

    # save
    if saveplots:
        figname = os.path.join(figdir, dn, 'trial{}.png'.format(i))
        plt.savefig(figname)

    plt.pause(.25)


# maybe show latent vs active units?

# %% make gifs
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

savegif = True

lr_clusters = .1
lr_clusters_group = .5
upd1noise = .09  # .1/.2
recnoise = 0.035  # atm, 0 for dupd, .01 for catlearn

# double update
dn = 'demos_dupd_{}units_k{}_lr{}_grouplr{}_upd1noise{}_recnoise{}'.format(
      n_units, k, lr_clusters, lr_clusters_group, upd1noise, recnoise)

# # catlearn
# dn = 'demos_catlearn_{}units_k{}_lr{}_grouplr{}_upd1noise{}_recnoise{}'.format(
#       n_units, k, lr_clusters, lr_clusters_group, upd1noise, recnoise)

images = []
for i in range(20):
    fname = os.path.join(figdir, dn, 'trial{}.png'.format(i))
    images.append(imageio.imread(fname))

if savegif:
    imageio.mimsave(
        os.path.join(figdir, dn, 'trials.gif'), images, duration=.4)


# %% plot laplacians (attention weights) for model schematic
# - to get hex:
# import matplotlib.colors as mcols
# mcols.to_hex(col)

saveplots = True

linewidth = 20  # has to be thicker coz small in figure
ylims = (0, .35)

# higher variance
col = 'cornflowerblue'  # '#6495ed'

col = 'mediumslateblue'  # purple - '#7b68ee'
col = 'mediumpurple'  # '#9370db'
col = 'violet'  # pink - '#ee82ee'
col = 'pink'  # '#ffc0cb'
beta = 4.
x = np.arange(-5, 5.1, .1)
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.plot(x, 1/(2 * beta) * (np.exp(-np.abs(x) / beta)),
        linewidth=linewidth, color=col)
ax.set_ylim(ylims)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
if saveplots:
    figname = os.path.join(figdir,
                           'attn_laplace_beta{}_{}.pdf'.format(beta, col))
    plt.savefig(figname)
plt.show()

# lower variance
col = 'limegreen'  # '#32cd32'
col = 'mediumseagreen'  # '#3cb371'
col = 'orange'  # '#ffa500'
col = 'sandybrown'  # '#f4a460'
beta = 1.5
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.plot(x, 1/(2 * beta) * (np.exp(-np.abs(x) / beta)),
        linewidth=linewidth, color=col)
ax.set_ylim(ylims)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
if saveplots:
    figname = os.path.join(figdir,
                           'attn_laplace_beta{}_{}.pdf'.format(beta, col))
    plt.savefig(figname)
plt.show()


