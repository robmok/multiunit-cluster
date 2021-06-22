#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:36:24 2021

@author: robert.mok
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import itertools as it
# from scipy.stats import norm
from scipy.stats import binned_statistic_dd

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
import scores   # grid cell scorer from Banino
from scipy.ndimage.filters import gaussian_filter

from MultiUnitCluster import (MultiUnitCluster, train_unsupervised,
                              train_unsupervised_simple)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')


# functions for spatial simulations, grid scores
def generate_path(n_trials, n_dims, seed=None):

    if seed:
        torch.manual_seed(seed)

    step_set = [-.1, -.075, -.05, -.025, 0, .025, .05, .075, .1]
    # step_set = [-.075, -.05, -.025, 0, .025, .05, .075]
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

# %% unsupervised

n_dims = 2
n_epochs = 1
n_trials = 50000
attn_type = 'dimensional_local'

# generate path
path = generate_path(n_trials, n_dims)

# # Plot the path
# start = path[:1]
# stop = path[-1:]
# fig = plt.figure(figsize=(8, 8), dpi=200)
# ax = fig.add_subplot(111)
# ax.scatter(path[:, 0], path[:, 1], c='blue', alpha=0.5, s=0.1)
# ax.plot(path[:, 0], path[:, 1], c='blue', alpha=0.75, lw=0.25, ls='-')
# ax.plot(start[:, 0], start[:, 1], c='red', marker='+')
# ax.plot(stop[:, 0], stop[:, 1], c='black', marker='o')
# plt.title('2D Random Walk')
# plt.tight_layout(pad=0)

n_units = 1000
k = .01

# annealed lr
orig_lr = .3
ann_c = (1/n_trials)/n_trials # 1/annC*nBatch = nBatch: constant to calc 1/annEpsDecay
ann_decay = ann_c * (n_trials * 100)  # 100
lr = [orig_lr / (1 + (ann_decay * itrial)) for itrial in range(n_trials)]
# plt.plot(torch.tensor(lr))
# plt.show()

# annealing for 2nd update
lr_group = np.array(lr) * 1.5 # 1.5, max 2. if too high will go toward centre.

# fixed thresh: .7

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': 1.2,  # low for larger/fewer fields, high for smaller/more fields.
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 1,  # response parameter, non-negative
    'lr_attn': .0,
    'lr_nn': .0,
    'lr_clusters':  lr,  # .01,
    'lr_clusters_group': lr_group,  # .1 w/out noise (.15 ok. .2 too much)
    'k': k
    }

noise = None
noise = {'update1': [0, .025],  # unit position updates 1 & 2
         'update2': [0, .025],  # no noise here also makes sense - since there is noise in 1 and you get all that info.
         'recruit': [0., .025],  # recruitment position placement
         }

# model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)
# train_unsupervised(model, torch.tensor(path, dtype=torch.float32), n_epochs)

# batch training
# for batch, c needs to be higher or thresh lower
batch_size = 200 #  n_trials * .005
nbatch = int(n_trials // batch_size)

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)

for ibatch in range(nbatch):

    batch_trials = [int(batch_size * (ibatch)),
                    int((batch_size * ibatch) + batch_size)]
    inputs = torch.tensor(path[batch_trials[0]:batch_trials[1]],
                          dtype=torch.float32)

    train_unsupervised(model, inputs, n_epochs, batch_upd=ibatch, noise=noise)

    print(len(model.recruit_units_trl))

# % plot

results = torch.stack(model.units_pos_trace, dim=0)

# group
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
# ax.scatter(results[-1, :, 0], results[-1, :, 1])
ax.scatter(results[-1, model.active_units, 0],
            results[-1, model.active_units, 1])
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()

# over time
plot_trials = torch.tensor(torch.linspace(0, nbatch * n_epochs, 20),
                            dtype=torch.long)

# plot_trials = torch.arange(10)

for i in plot_trials[0:-1]:
    plt.scatter(results[i, model.active_units, 0],
                results[i, model.active_units, 1])
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.pause(.5)

# grid computations

# # plot activations during training
# # plot from trial n to ntrials
# # plot_trials = [0, n_trials-1]
# plot_trials = [int(n_trials//1.5), n_trials-1]

# # get saved activations
# act = torch.stack(
#     model.fc1_act_trace)[plot_trials[0]:plot_trials[1]-1].sum(axis=1).detach()

# nbins = 40
# act_map = _compute_activation_map(
#     path[plot_trials[0]:plot_trials[1]-1], act, nbins, statistic='sum')

# # normalize by times visited the location
# norm_mat = normalise_act_map(nbins, act_map.binnumber)

# # plot normalized act_map
# ind = np.nonzero(norm_mat)
# act_map_norm = act_map.statistic.copy()
# act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]
# # plt.imshow(act_map.statistic,
# #            vmin=np.percentile(act_map.statistic, 1),
# #            vmax=np.percentile(act_map.statistic, 99))
# # plt.show()
# plt.imshow(act_map_norm,
#            vmin=np.percentile(act_map_norm, 1),
#            vmax=np.percentile(act_map_norm, 99))
# plt.show()

# plot activation after training - unit positions at the end, fixed
# generate new test path
n_trials_test = n_trials // 2
path_test = generate_path(n_trials_test, n_dims, seed=None)

# get act
nbins = 40  # TODO - check why > 40 bins then get weird crisscross patterns
act_test = []
for itrial in range(n_trials_test):
    if np.mod(itrial, 1000) == 0:
        print(itrial)
    dim_dist = abs(path_test[itrial] - model.units_pos)
    dist = _compute_dist(dim_dist, model.attn, model.params['r'])
    act = _compute_act(dist, model.params['c'], model.params['p'])
    act[~model.active_units] = 0  # not connected, no act
    _, win_ind = torch.topk(act,
                            int(model.n_units * model.params['k']))
    act_test.append(act[win_ind].sum().detach())

act_map = _compute_activation_map(
    path_test, torch.tensor(act_test), nbins, statistic='sum')

# normalize by times visited the location
norm_mat = normalise_act_map(nbins, act_map.binnumber)

# plot normalized act_map
ind = np.nonzero(norm_mat)
act_map_norm = act_map.statistic.copy()
act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]
# plt.imshow(act_map.statistic,
#            vmin=np.percentile(act_map.statistic, 1),
#            vmax=np.percentile(act_map.statistic, 99))
# plt.show()
plt.imshow(act_map_norm,
           vmin=np.percentile(act_map_norm, 1),
           vmax=np.percentile(act_map_norm, 99))
plt.show()

# compute grid scores
score_60, score_90, _, _, sac = _compute_grid_scores(act_map_norm)

print(score_60, score_90)

# autocorrelogram
plt.imshow(sac)
plt.show()
# %% unsupervised simple (no recruitment)

"""
#  k is key for number of virtual clusters that come up. smaller k = more; larger k = fewer clusters 
# lr_group has to be large-ish, else virtual clusters don't form (scattered).
# lr_group has to be > lr_clusters, else virtual cluster don't form. but not too high else clusters go toward centre

# - i think the learning rates might lead to more/less grid like patterns - check which matters more (can use banino's grid code)
# - need reduction of lr over time?
"""
# To check
# - one thing i see from plotting over time is that clusters change sometimes change across virtual clusters. need lower lr?
# looks like less later on though. maybe ok?

n_dims = 2
n_epochs = 1
n_trials = 500000
attn_type = 'dimensional_local'

# generate path
path = generate_path(n_trials, n_dims)

# random numbers - not a path
path = torch.tensor(np.around(np.random.rand(n_trials, n_dims), decimals=3))


# n_units=1k, k=.05, orig_lr=.2, group=.85 works (20 clusters)
# n_units=1k, k=.1/.2, orig_lr=.1, group=.65/85 works (10 clusters), 10k trials
# - when k>.13 not working.. group needs to be >.9, lr_clus needs to be tiny.
# e.g. k=.15, orig_lr=.005 looks ok. potential prob - need diff lrs for different ks?

# try smaller k with lower lrs - might work; just more trials
# - yep k=.1 still ok w orig_lr=.005. - needs 50k trials now. (20 not enough)
# - as with k=.15, seems like clusters are more central, then expands out
# I guess this is coz the 2nd update pulls them in, then later when virtual
# clusters form, the 1st lr over takes

# - ok now 50k trials doens't work for k=.15. too much learning at the start now?

n_units = 1000
k = .15

# annealed lr
orig_lr = .005
ann_c = (1/n_trials)/n_trials
ann_decay = ann_c * (n_trials * 100)  # 100
lr = [orig_lr / (1 + (ann_decay * itrial)) for itrial in range(n_trials)]
plt.plot(lr)

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': 1.2,
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 1,  # response parameter, non-negative
    'lr_attn': .0,
    'lr_nn': .25,
    'lr_clusters': lr,  # np.array(lr) * 0 + .001,
    'lr_clusters_group': .95,
    'k': k
    }

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)

train_unsupervised_simple(model, path, n_epochs)

results = torch.stack(model.units_pos_trace, dim=0)

# group
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.scatter(results[-1, :, 0], results[-1, :, 1])
# ax.scatter(results[-1, model.active_units, 0],
#             results[-1, model.active_units, 1])
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()

# over time
plot_trials = torch.tensor(torch.linspace(0, n_trials, 10),
                            dtype=torch.long)

for i in plot_trials[0:-1]:

    plt.scatter(results[i, :, 0],
                results[i, :, 1])
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.pause(.5)


# n_trials_test = int(n_trials * .5)
# # path_test = generate_path(n_trials_test, n_dims, seed=None)
# path_test = torch.tensor(
#     np.around(np.random.rand(n_trials_test, n_dims), decimals=3))

# get act
nbins = 40
act_test = []
for itrial in range(n_trials_test):
    if np.mod(itrial, 1000) == 0:
        print(itrial)
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

# plot normalized act_map
ind = np.nonzero(norm_mat)
act_map_norm = act_map.statistic.copy()
act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]
plt.imshow(act_map_norm,
            vmin=np.percentile(act_map_norm, 1),
            vmax=np.percentile(act_map_norm, 99))
plt.show()

# compute grid scores
score_60, score_90, _, _, sac = _compute_grid_scores(act_map_norm)

print(score_60, score_90)

# autocorrelogram
plt.imshow(sac)
plt.show()

# %% run sims

save_sims = True

n_sims = 100

# model spec
n_dims = 2
n_epochs = 1
n_trials = 100000
attn_type = 'dimensional_local'

# run over different k values, n_units, c vals
# - can try different threshold values, but prob keep constant to show effects
# select threshold w nice fields (not too big/small fields and gd spacing)
n_units = 1000
k = .01

# batch params
batch_size = 500  # n_trials * .005
nbatch = int(n_trials // batch_size)

# thresh=.9
# - c=2/2.5, 3 fields. c=1.3-1.7, 4-7 fields. c=1.2, 9-10. c=1, 30+

# re-run with new thresh

c_vals = [1.2, 1.6, 2.]
c_vals = [1.8]  # sim 6 started ~ 16:27. 77 at 17:31. slightly faster?

# annealed lr
orig_lr = .2
# 1/ann_c*nbatch=nbatch: constant to calc 1/ann_decay
ann_c = (1/n_trials)/n_trials
ann_decay = ann_c * (n_trials * 100)  # 100
lr = [orig_lr / (1 + (ann_decay * i)) for i in range(n_trials)]
# plt.plot(torch.tensor(lr))
# plt.show()

# annealed for 2nd update
anneal_lr_group = True
if anneal_lr_group:
    lr_group = np.array(lr) * 2

# orig_lr = .0001
# # 1/annC*nBatch = nBatch: constant to calc 1/annEpsDecay
# ann_c = (1/n_trials)/n_trials
# ann_decay = ann_c * (n_trials * 20)
# lr_attn = [orig_lr / (1 + (ann_decay * i)) for i in range(n_trials)]

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': [],  # define in loop below
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 1,  # response parameter, non-negative
    'lr_attn': 0.,  # lr_attn,
    'lr_nn': .0,
    'lr_clusters': lr,  # annealed
    'lr_clusters_group': lr_group,
    'k': k
    }

# # dfs - gridscore, recruit n, seeds (in 1 df, load and save)
# wd = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/muc_results'
# fname1 = (
#     os.path.join(wd,'spatial_gscore_batch_ann_cvals_{}units_k{}_startlr{}_\
# grouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims.pkl'.format(n_units, params['k'],
# orig_lr,params['lr_clusters_group'], params['lr_attn'], n_trials, n_sims))
#     )
# fname2 = (
#     os.path.join(wd, 'spatial_recruit_batch_ann_cvals_{}units_k{}_startlr{}_\
# grouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims.pkl'.format(n_units, params['k'],
# orig_lr, params['lr_clusters_group'], params['lr_attn'], n_trials, n_sims))
#     )
# fname3 = (
#     os.path.join(wd, 'spatial_seeds_batch_ann_cvals_{}units_k{}_startlr{}_\
# grouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims.pkl'.format(n_units, params['k'],
# orig_lr, params['lr_clusters_group'], params['lr_attn'], n_trials, n_sims))
#     )
# # clus positions, activation map
# fname4 = (
#     os.path.join(wd, 'spatial_actmapclus_batch_ann_cvals_{}units_k{}_startlr\
# {}_grouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims'.format(n_units, params['k'],
# orig_lr, params['lr_clusters_group'], params['lr_attn'], n_trials, n_sims))
#     )

# if annealed lr group

wd = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/muc_results'
fname1 = (
    os.path.join(wd,'spatial_gscore_batch{}_ann_cvals_{}units_k{}_startlr{}_\
startgrouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims.pkl'.format(batch_size, n_units, params['k'],
orig_lr, lr_group[0], params['lr_attn'], n_trials, n_sims))
    )
fname2 = (
    os.path.join(wd, 'spatial_recruit_batch{}_ann_cvals_{}units_k{}_startlr{}_\
startgrouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims.pkl'.format(batch_size, n_units, params['k'],
orig_lr, lr_group[0], params['lr_attn'], n_trials, n_sims))
    )
fname3 = (
    os.path.join(wd, 'spatial_seeds_batch{}_ann_cvals_{}units_k{}_startlr{}_\
startgrouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims.pkl'.format(batch_size, n_units, params['k'],
orig_lr, lr_group[0], params['lr_attn'], n_trials, n_sims))
    )
# clus positions, activation map
fname4 = (
    os.path.join(wd, 'spatial_actmapclus_batch{}_ann_cvals_{}units_k{}_startlr\
{}startgrouplr{}_attnlr{}_thresh.7_{}ktrls_{}sims'.format(batch_size, n_units, params['k'],
orig_lr, lr_group[0], params['lr_attn'], n_trials, n_sims))
    ) 

# load and add to sims (if True) or make new files (if False)
load = True
if load:
    df_gscore = pd.read_pickle(fname1)
    df_recruit = pd.read_pickle(fname2)
    df_seeds = pd.read_pickle(fname3)

    # add c_vals conditions to ech df
    [df_gscore.insert(len(df_gscore.columns), i,
                      pd.Series(np.zeros(len(df_gscore)),
                                index=df_gscore.index))
     for i in c_vals]
    [df_recruit.insert(len(df_recruit.columns), i,
                       pd.Series(np.zeros(len(df_recruit)),
                                 index=df_recruit.index))
     for i in c_vals]
    [df_seeds.insert(len(df_seeds.columns), i,
                     pd.Series(np.zeros(len(df_seeds)),
                               index=df_seeds.index))
     for i in c_vals]

else:  # new files - will overwrite files if exist

    df_gscore = pd.DataFrame(columns=c_vals, index=range(n_sims))
    df_recruit = pd.DataFrame(columns=c_vals, index=range(n_sims))
    df_seeds = pd.DataFrame(columns=c_vals, index=range(n_sims))

# start
for c in c_vals:
    shuffle_seeds = torch.randperm(n_sims*100)[:n_sims]
    score_60 = []
    n_recruit = []
    pos_trace = []
    act_map_all = []

    for isim in range(n_sims):

        print('sim {}'.format(isim))

        # params to change over loops
        params['c'] = c

        # generate path
        path = generate_path(n_trials, n_dims, seed=shuffle_seeds[isim])

        # train model
        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)

        for ibatch in range(nbatch):

            batch_trials = [int(batch_size * (ibatch)),
                            int((batch_size * ibatch) + batch_size)]
            inputs = path[batch_trials[0]:batch_trials[1]]

            train_unsupervised(model, inputs, n_epochs, batch_upd=ibatch)

        print(len(model.recruit_units_trl))

        # generate new test path
        nbins = 40
        n_trials_test = int(n_trials * .5)  # .5 for 50k trials.
        path_test = generate_path(n_trials_test, n_dims, seed=None)
        act_test = []
        for itrial in range(n_trials_test):
            dim_dist = abs(path_test[itrial] - model.units_pos)
            dist = _compute_dist(dim_dist, model.attn, model.params['r'])
            act = _compute_act(dist, model.params['c'], model.params['p'])
            act[~model.active_units] = 0  # not connected, no act
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            act_test.append(act[win_ind].sum().detach())
        act_map = _compute_activation_map(
            path_test, torch.tensor(act_test), nbins, statistic='sum')
        norm_mat = normalise_act_map(nbins, act_map.binnumber)
        # normalized act_map
        ind = np.nonzero(norm_mat)
        act_map_norm = act_map.statistic.copy()
        act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]

        # compute grid scores
        score_60_, score_90_, _, _, sac = _compute_grid_scores(act_map_norm)
        
        print(score_60_)

        # save stuff
        score_60.append(score_60_)
        n_recruit.append(len(model.recruit_units_trl))
        pos_trace.append(model.units_pos_trace)
        act_map_all.append(act_map_norm)

    df_gscore[c] = np.array(score_60)
    df_recruit[c] = np.array(n_recruit)
    df_seeds[c] = np.array(shuffle_seeds)

    # save per c value (else too big) - unit pos, act map (no act_trace - huge)
    if save_sims:
        fname_pt = fname4 + '_c{}.pt'.format(c)
        torch.save({"pos": pos_trace,
                    "act_map": act_map_all},
                   fname_pt)
# save df
if save_sims:
    df_gscore.to_pickle(fname1)
    df_recruit.to_pickle(fname2)
    df_seeds.to_pickle(fname3)

# %%

saveplots = False

# load dfs
df_gscore = pd.read_pickle(fname1)
df_recruit = pd.read_pickle(fname2)
df_seeds = pd.read_pickle(fname3)

# load actmap for specific c values
c = 1.2
fname_pt = fname4 + '_c{}.pt'.format(c)
f = torch.load(fname_pt)
act_maps = f['act_map']


df_gscore.hist(bins=20)
if saveplots:
    figname = (
        os.path.join(figdir, 'spatial_gscore_hist_{}units_k{}_startlr{}_grouplr{}_thresh.7_{}trls.pdf'.format(
            n_units, k, orig_lr, params['lr_clusters_group'], n_trials))
    )
    plt.savefig(figname)
plt.show()

df_recruit.hist()
if saveplots:
    figname = (
        os.path.join(figdir, 'spatial_recruit_hist_{}units_k{}_startlr{}_grouplr{}_thresh.7_{}trls.pdf'.format(
            n_units, k, orig_lr, params['lr_clusters_group'], n_trials))
    )
    plt.savefig(figname)
plt.show()


# act_map and autocorrelogram
isim = 0

_, _, _, _, sac = _compute_grid_scores(act_map_all[isim])


fig, ax = plt.subplots(1, 2)
ax[0].imshow(act_map_all[isim])
ax[0].set_title('c = {}'.format(c))
ax[1].imshow(sac)
ax[1].set_title('g = {}'.format(np.around(df_gscore[c][isim], decimals=3)))
if saveplots:
    figname = (
        os.path.join(figdir, 'spatial_actmap_xcorr_c{}_{}units_k{}_startlr{}_grouplr{}_thresh.7_{}trls_sim{}.pdf'.format(
            c, n_units, k, orig_lr, params['lr_clusters_group'], n_trials, isim))
    )
    plt.savefig(figname)
plt.show()
    
    

# %%

# run for different learning rates for lr_clusters and lr_group
# lr_clusters = torch.linspace(.001, .5, 10)
# lr_group = torch.linspace(.1, 2, 10)

# lr_clusters = torch.arange(.001, .5, .05)
# lr_group = torch.arange(.1, 2, .2)

# results = torch.zeros(n_units, n_dims, len(lr_clusters), len(lr_group))
# for i, j in it.product(range(len(lr_clusters)), range(len(lr_group))):
#     params['lr_clusters'] = lr_clusters[i]
#     params['lr_clusters_group'] = lr_group[j]
#     model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)
#     train_unsupervised(model, inputs, n_epochs)
#     results[:, :, i, j] = torch.stack(model.units_pos_trace, dim=0)[-1]


# # fig, ax = plt.subplots(len(lr_clusters), len(lr_group))
# # for i, j in it.product(range(len(lr_clusters)), range(len(lr_group))):
# #     ax[i, j].scatter(results[:, 0, i, j], results[:, 1, i, j], s=.005)
# #     ax[i, j].set_xlim([0, 1])
# #     ax[i, j].set_ylim([0, 1])


# wd = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/multiunit-cluster_figs'

# lr = lr_group[3]
# j = torch.nonzero(lr_group == lr)
# for i in range(len(lr_clusters)):
#     plt.scatter(results[:, 0, i, j], results[:, 1, i, j])
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.show()

#     # figname = os.path.join(wd,
#     #                         'hipp_cluster_across_lrclus' +
#     #                         str(round(lr_clusters[i].tolist(), 3)) +
#     #                         '_lrgroup' + str(round(lr.tolist(), 3)) + '.png')
#     # plt.savefig(figname)
#     # plt.show()


# # lr = lr_clusters[5]  # >.1 [3/4/5]
# # i = torch.nonzero(lr_clusters == lr)
# # for j in range(len(lr_group)):
# #     plt.scatter(results[:, 0, i, j], results[:, 1, i, j])
# #     plt.xlim([0, 1])
# #     plt.ylim([0, 1])
# #     plt.show()

#     # figname = os.path.join(wd,
#     #                        'hipp_cluster_across_lrgroup' +
#     #                        str(round(lr_group[j].tolist(), 3)) +
#     #                        '_lrclus' +
#     #                        str(round(lr.tolist(), 3)) + '.png')
#     # plt.savefig(figname)
#     # plt.show()
