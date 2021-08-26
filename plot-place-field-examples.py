#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:26:09 2021

Example place fields

- currently just copy and pasted stuff from sim
- can make it 'simpler' - just get activations of the gaussian...

@author: robert.mok
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
import itertools as it

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

n_dims = 2
nbins = 40


def _compute_dist(dim_dist, attn_w, r):
    # d = torch.sum(attn_w * (dim_dist**r), axis=1) ** (1/r)
    d = torch.sum(attn_w * (dim_dist**r)) ** (1/r)
    return d


def _compute_act(dist, c, p):
    return c * torch.exp(-c * dist)  # sustain-like


def _compute_activation_map(
        pos, activations, nbins, statistic='sum'):
    return binned_statistic_dd(
        pos,
        activations,
        bins=nbins,
        statistic=statistic,
        range=np.array(np.tile([0, 1], (n_dims, 1))),
        expand_binnumbers=True)  # added for normalization


def normalise_act_map(nbins, binnumber):
    """ binnumber is act_map.binnumber from binned_statistic_dd
    note: expand_binnumbers=True when calling binned_statistic_dd
    """
    norm_mat = np.zeros([nbins, nbins])
    coord = np.array(list(it.product(range(nbins),
                                     range(nbins))))
    for x in coord:
        norm_mat[x[0], x[1]] = (
            np.sum((x[0] == binnumber[0, :]-1)  # bins start from 1
                   & (x[1] == binnumber[1, :]-1))
            )
    return norm_mat


# %%

saveplots = False

act_func = 'gauss'  # laplace or gauss

# centres
loc = [.25, .25]
loc = [.5, .25]
loc = [.75, .75]
loc = [.25, .75]

if act_func == 'gauss':

    # wide tuning
    var = .075
    cmap = 'Blues'

    # narrow tuning
    var = .005
    cmap = 'YlOrBr'

    cov = torch.cholesky(torch.eye(2) * var)
    mvn1 = torch.distributions.MultivariateNormal(torch.tensor(loc),
                                                  scale_tril=cov)

else:  # laplace
    loc0 = [.25, .25]
    loc1 = [.75, .75]

    # wide tuning
    c = .5
    cmap = 'Blues'
    # cmap = 'PuBu'

    # narrow tuning
    c = 20
    cmap = 'YlOrBr'


path_test = torch.tensor(
    list(it.product(torch.arange(0, 1, .01), torch.arange(0, 1, .01))))

nbins = 40
act_test = []
for itrial in range(len(path_test)):

    # gaussian
    if act_func == 'gauss':
        act = torch.exp(mvn1.log_prob(path_test[itrial].detach()))
    elif act_func == 'laplace':
        # laplacian / peaked exponential
        dim_dist = abs(path_test[itrial].detach() - torch.tensor(loc0))
        dist = _compute_dist(dim_dist, torch.tensor([.5, .5]), 1)
        act0 = _compute_act(dist, c, 1)

        dim_dist = abs(path_test[itrial].detach() - torch.tensor(loc1))
        dist = _compute_dist(dim_dist, torch.tensor([.5, .5]), 1)
        act1 = _compute_act(dist, c, 1)

        if act0 > act1:
            act = act0
        else:
            act = act1

    act_test.append(act.detach())
act_map = _compute_activation_map(
    path_test, torch.tensor(act_test), nbins, statistic='sum')
norm_mat = normalise_act_map(nbins, act_map.binnumber)

ind = np.nonzero(norm_mat)
act_map_norm = act_map.statistic.copy()
act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]

fig, ax = plt.subplots()
ax.imshow(act_map_norm, cmap=cmap)
# ax.set_title('k = {}'.format(k))
ax.set_xticks([])
ax.set_yticks([])
if saveplots:
    if act_func == 'gauss':
        fn = 'place_field_example_loc{:.2f}-{:.2f}_cov{:.3f}'.format(
            loc[0], loc[1], cov[0, 0])
    elif act_func == 'laplace':
        fn = 'laplace_act_loc{:.2f}-{:.2f}-{:.2f}-{:.2f}_c{:.3f}'.format(
            loc0[0], loc0[1], loc1[0], loc1[1], c)

    figname = os.path.join(figdir, 'actmaps/' + fn)
    # plt.savefig(figname, dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()
