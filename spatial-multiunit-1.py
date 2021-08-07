#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 00:06:10 2021

@author: robert.mok
"""
import os
import sys
import numpy as np
# import pandas as pd
import torch
# import matplotlib.pyplot as plt
import itertools as it
# from scipy.stats import norm
from scipy.stats import binned_statistic_dd
# import seaborn as sns
import time

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
import scores   # grid cell scorer from Banino
from scipy.ndimage.filters import gaussian_filter

from MultiUnitCluster import (MultiUnitCluster, train_unsupervised_simple)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')
wd = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/muc_results'


# functions for spatial simulations, grid scores
def generate_path(n_trials, n_dims, seed=None):

    if seed:
        torch.manual_seed(seed)

    # step_set = [-.1, -.075, -.05, -.025, 0, .025, .05, .075, .1]
    step_set = [-.075, -.05, -.025, 0, .025, .05, .075]
    path = np.zeros([n_trials, n_dims])
    path[0] = np.around(np.random.rand(2), decimals=3)  # origin
    for itrial in range(1, n_trials):
        step = np.random.choice(a=step_set, size=n_dims)  # 1 trial at a time
        # only allow 0 < steps < 1
        while (np.any(path[itrial-1] + step < 0)
               or np.any(path[itrial-1] + step > 1)):
            step = np.random.choice(a=step_set, size=n_dims)
        path[itrial] = path[itrial-1] + step
    return torch.tensor(path, dtype=torch.float32)


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


def _compute_grid_scores(activation_map, smooth=False):
    # n_dims = len(activation_map.shape)
    if smooth:
        activation_map = gaussian_filter(activation_map, sigma=.8)
    # mask parameters
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    scorer = scores.GridScorer(
        len(activation_map), [0, len(activation_map)-1], masks_parameters)
    score_60, score_90, max_60_mask, max_90_mask, sac = scorer.get_scores(
        activation_map)
    return score_60, score_90, max_60_mask, max_90_mask, sac


def _compute_dist(dim_dist, attn_w, r):
    # since sqrt of 0 returns nan for gradient, need this bit
    # e.g. euclid, can't **(1/2)
    if r > 1:
        d = torch.zeros(len(dim_dist))
        ind = torch.sum(dim_dist, axis=1) > 0
        dim_dist_tmp = dim_dist[ind]
        d[ind] = torch.sum(attn_w * (dim_dist_tmp ** r), axis=1)**(1/r)
    else:
        d = torch.sum(attn_w * (dim_dist**r), axis=1) ** (1/r)
    return d


def _compute_act(dist, c, p):
    """ c = 1  # ALCOVE - specificity of the node - free param
        p = 2  # p=1 exp, p=2 gauss
    """
    # return torch.exp(-c * (dist**p))
    return c * torch.exp(-c * dist)  # sustain-like


# %% unsupervised simple (no recruitment)

save_sims = True

n_dims = 2
n_epochs = 1
n_trials = 500000
attn_type = 'dimensional_local'

# params to test
# k:  .08 (12 clus), .1 (9), .13 (7), .26 (5), .28 (3)
# orig_lr: .001, .0025, .005
# lr_group: .5, .85, 1

params = [[.08, .1, .13, .26, .28],
          [.001, .0025, .005],
          [.5, .85, 1]]

param_sets = torch.tensor(list(it.product(*params)))

n_units = 1000

n_sims = 100

for iset, p in enumerate(param_sets):

    # shuffle_seeds = torch.randperm(n_sims*100)[:n_sims]
    score_60 = []
    pos_trace = []
    act_map_all = []

    print('Running param set {} / {}'.format(iset, len(param_sets)))
    t0 = time.time()

    for isim in range(n_sims):

        print('sim {}'.format(isim))

        # generate path
        path = generate_path(n_trials, n_dims)

        k = p[0]
        # annealed lr
        orig_lr = p[1]
        ann_c = (1/n_trials)/n_trials
        ann_decay = ann_c * (n_trials * 100)  # 100
        lr = [orig_lr / (1 + (ann_decay * itrial))
              for itrial in range(n_trials)]

        lr_group = p[2]

        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': 1.2,
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1,  # response parameter, non-negative
            'lr_attn': .0,
            'lr_nn': .25,
            'lr_clusters': lr,  # np.array(lr) * 0 + .001,
            'lr_clusters_group': lr_group,
            'k': k
            }

        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)

        train_unsupervised_simple(model, path, n_epochs)

        # grid score
        n_trials_test = int(n_trials * .25)
        path_test = torch.tensor(
            np.around(np.random.rand(n_trials_test, n_dims), decimals=3))

        # get act
        nbins = 40
        act_test = []
        for itrial in range(n_trials_test):
            dim_dist = abs(path_test[itrial] - model.units_pos)
            dist = _compute_dist(dim_dist, model.attn, model.params['r'])
            act = _compute_act(dist, model.params['c'], model.params['p'])
            # act[~model.active_units] = 0  # not connected, no act
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            act_test.append(act[win_ind].sum().detach())
        act_map = _compute_activation_map(
            path_test, torch.tensor(act_test), nbins, statistic='sum')
        norm_mat = normalise_act_map(nbins, act_map.binnumber)

        # get normalized act_map
        ind = np.nonzero(norm_mat)
        act_map_norm = act_map.statistic.copy()
        act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]

        # compute grid scores
        score_60_, _, _, _, sac = _compute_grid_scores(act_map_norm)

        # save stuff
        score_60.append(score_60_)
        # get just 100 trials for now. can increase later
        pos_trace_tmp = [model.units_pos_trace[i]
                         for i in np.arange(0, n_trials, 5000)]
        pos_trace.append(pos_trace_tmp)
        act_map_all.append(act_map_norm)

    if save_sims:

        fn = (
            os.path.join(wd, 'spatial_simple_ann_{:d}units_k{:.2f}_'
                         'startlr{:.3f}_grouplr{:.3f}_{:d}ktrls_'
                         '{:d}sims.pkl'.format(
                             n_units, p[0], p[1], p[2], n_trials//1000, n_sims))
            )
        # open_file = open(fn1, "wb")
        # pickle.dump(score_60, open_file)
        # open_file.close()

        # unit pos, act map (no act_trace - huge)
        torch.save({"gscore": score_60,
                    "pos": pos_trace,
                    "act_map": act_map_all},
                   fn)
    t1 = time.time()
    print(t1-t0)

# %%


# # group
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111)
# ax.scatter(results[-1, :, 0], results[-1, :, 1])
# # ax.scatter(results[-1, model.active_units, 0],
# #             results[-1, model.active_units, 1])
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# plt.show()




# # over time
# plot_trials = torch.tensor(torch.linspace(0, n_trials, 10),
#                            dtype=torch.long)

# for i in plot_trials[0:-1]:  # range(20):  #

#     plt.scatter(results[i, :, 0],
#                 results[i, :, 1])
#     plt.xlim([-.05, 1.05])
#     plt.ylim([-.05, 1.05])
#     plt.pause(.5)

# # autocorrelogram
# plt.imshow(sac)
# plt.show()




# plt.imshow(act_map_norm,
#             vmin=np.percentile(act_map_norm, 1),
#             vmax=np.percentile(act_map_norm, 99))
# plt.show()