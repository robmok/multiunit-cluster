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

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

# %% double update demo

# one 2D gaussian - k units update

# sample from a 2D gaussian
mu1 = np.array([.25, .75])
var1 = np.array([.005, .005])
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

# 3d plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, rv1.pdf(pos), cmap='Greys', linewidth=0)
ax.grid(False)
ax.set_zticks([])
plt.show()

# assign label - just one 'category'
inputs = torch.tensor(x1, dtype=torch.float32)
output = torch.zeros(npoints, dtype=torch.long)

n_epochs = 1

# double update demo

# model
attn_type = 'dimensional_local'
n_units = 500
n_dims = inputs.shape[1]
loss_type = 'cross_entropy'
k = .05

# scaling lr
lr_scale = (n_units * k) / 1

params = {
    'r': 1,
    'c': 1.,
    'p': 1,
    'phi': 2.5,
    'beta': 1.,
    'lr_attn': .0,
    'lr_nn': .1/lr_scale,
    'lr_clusters': .1,
    'lr_clusters_group': .15,
    'k': k
    }

lesions = None

# noise - mean and sd of noise to be added
noise = None
noise = {'update1': [0, .15],  # unit position updates 1 & 2
         'update2': [0, .0],  # no noise here also makes sense
         'recruit': [0., .05],  # recruitment position placement
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
this. lr_clusters = .05, lr_group = .1, pretty good. no lr_group means units
keep going around, whereas lr_group model settles to the centre pretty quickly

"""

# plot both updates

# have each trial twice - to plot with double update with same stim
inputs_d = torch.stack([np.repeat(inputs[:, 0], 2),
                        np.repeat(inputs[:, 1], 2)]).T

results = torch.stack(
    model.units_pos_bothupd_trace, dim=0)[:, model.active_units]

# plot_trials = torch.tensor(torch.linspace(0, len(inputs) * n_epochs, 10),
#                             dtype=torch.long)

plot_trials = torch.arange(50)

for i in plot_trials[0:-1]:

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    # plot distribution stimuli come from (2d gaussian)
    ax.contour(x, y, rv1.pdf(pos), cmap='Greys', alpha=.35)

    # stimulus pos on trial i
    ax.scatter(inputs_d[i, 0],
               inputs_d[i, 1], c='black', marker='x', s=25, linewidth=1.2)

    # unit pos on trial i - showing double update
    ax.scatter(results[i, :, 0], results[i, :, 1], s=8)

    ax.set_facecolor((.8, .8, .8))
    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])
    ax.set_aspect('equal', adjustable='box')

    plt.pause(.25)


# %% concept learning toy example

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
npoints = 200
x1 = np.random.multivariate_normal(
    [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)
x2 = np.random.multivariate_normal(
    [mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]], npoints)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax1.contour(x, y, rv1.pdf(pos), cmap='Blues')
ax1.contour(x, y, rv2.pdf(pos), cmap='Greens')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_facecolor((.5, .5, .5))
ax1.scatter(x1[:, 0], x1[:, 1], marker='x', c='blue', s=7, linewidth=.75)
ax1.scatter(x2[:, 0], x2[:, 1], marker='x', c='green', s=7, linewidth=.75)
ax1.set_aspect('equal', adjustable='box')
plt.show()

# 3d plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x, y, rv1.pdf(pos), cmap='GnBu', alpha=.75)
ax.plot_surface(x, y, rv2.pdf(pos), cmap='RdPu', alpha=.5)
ax.grid(False)
ax.set_zticks([])
plt.show()

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
    'c': .75,
    'p': 1,
    'phi': 2.5,
    'beta': 1.,
    'lr_attn': .0,
    'lr_nn': .1/lr_scale,
    'lr_clusters': .075,
    'lr_clusters_group': .1,
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

for i in plot_trials[0:-1]:

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    # plot distribution stimuli come from (2d gaussian)
    ax.contour(x, y, rv1.pdf(pos), cmap='Blues', alpha=.25)
    ax.contour(x, y, rv2.pdf(pos), cmap='Greens', alpha=.25)

    # stimulus pos on trial i
    ax.scatter(inputs_d[i, 0],
               inputs_d[i, 1], c='black', marker='x', s=25, linewidth=1.2)

    # unit pos on trial i - showing double update
    ax.scatter(results[i, :, 0], results[i, :, 1], c='grey', s=8)

    ax.set_facecolor((.8, .8, .8))
    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])
    ax.set_aspect('equal', adjustable='box')

    plt.pause(.25)


# maybe show latent vs active units?
