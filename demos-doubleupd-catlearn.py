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
# - maybe show 'latent' and active units?


# sample from a 2D gaussian
mu1 = np.array([.25, .75])
var1 = np.array([.005, .005])
cov1 = 0

x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
rv1 = multivariate_normal([mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]])

# % sampling
npoints = 200
x1 = np.random.multivariate_normal(
    [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax1.contour(x, y, rv1.pdf(pos), cmap='Blues')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_facecolor((.8, .8, .8))
ax1.scatter(x1[:, 0], x1[:, 1], c=(.4, .4, .4), s=7)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.gca().set_aspect('equal', adjustable='box')


# double update demo
aspect = 1
fig, ax = plt.subplots(1, 3)
for itrl in range(3):
    ax[itrl].scatter(x1[itrl, 0], x1[itrl, 1], s=7)
    ax[itrl].set_xlim([0, 1])
    ax[itrl].set_ylim([0, 1])
    ax[itrl].set_aspect(aspect)



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

# % sampling
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
ax1.scatter(x1[:, 0], x1[:, 1], c='blue', s=7)
ax1.scatter(x2[:, 0], x2[:, 1], c='green', s=7)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.gca().set_aspect('equal', adjustable='box')

# assign label (output)
inputs = torch.cat([torch.tensor(x1, dtype=torch.float32),
                    torch.tensor(x2, dtype=torch.float32)])
output = torch.cat([torch.zeros(npoints, dtype=torch.long),
                    torch.ones(npoints, dtype=torch.long)])







# recruit with noise - show double update


# update with noise - show double update




# maybe show latent vs active units?


