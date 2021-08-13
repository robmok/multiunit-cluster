#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:04:19 2021

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import itertools as it
from scipy.stats import binned_statistic_dd

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import imageio

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
import scores   # grid cell scorer from Banino
from scipy.ndimage.filters import gaussian_filter

from MultiUnitCluster import (MultiUnitCluster, train_unsupervised_k)

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


# %% plot

n_sims = 100
n_units = 1000
n_trials = 500000

# params to test
# k:  .08 (12 clus), .1 (9), .13 (7), .26 (5), .28 (3)
# orig_lr: .001, .0025, .005
# lr_group: .5, .85, 1

# params = [[.08, .1, .13, .26, .28],
#           [.001, .0025, .005, .0075],
#           [.85, 1]]

params = [[.08, .1, .13, .18, .28],
          [.0075, .01],
          [.6, .8, 1.]]

param_sets = torch.tensor(list(it.product(*params)))


# plot over k first
# - set lr's for now
lr = params[1][0]
lr_group = params[2][2]

df_gscore = pd.DataFrame(columns=params[0], index=range(n_sims))
for k in params[0]:

    p = [k, lr, lr_group]

    # load
    fn = (
        os.path.join(wd, 'spatial_simple_ann_{:d}units_k{:.2f}_'
                      'startlr{:.4f}_grouplr{:.3f}_{:d}ktrls_'
                      '{:d}sims.pkl'.format(
                          n_units, p[0], p[1], p[2], n_trials//1000, n_sims))
        )
    f = torch.load(fn)

    df_gscore[k] = np.array(f['gscore'])

# n gscores > threshold
# - turns out lr doesn't change this that much - mainly the magnitude changes
# if anything, slower has more...! faster has more high gscores maybe?
thr = .2
print('lr={}, lr_group={}: {} > {}'.format(
    lr, lr_group, (df_gscore > thr).sum().sum(), thr))

# gscore = f['gscore']
# pos_trace = f['pos']
# act_maps = f['act_map']


# sns.set(font_scale = 1.5)
# plt.style.use('seaborn-white')
fntsiz = 18

# gscore
g = sns.catplot(data=df_gscore, kind="violin")
sns.stripplot(color="k", alpha=0.2, size=3,
              data=df_gscore, ax=g.ax)

g.ax.set_ylim(-.75, 1.5)
# g.ax.set_xticklabels(g.ax.get_xmajorticklabels(), fontsize=fntsiz-3)
# g.ax.set_yticklabels(g.ax.get_ymajorticklabels(), fontsize=fntsiz-3)
g.ax.set_xlabel('k', fontsize=fntsiz)
g.ax.set_ylabel('Grid Score', fontsize=fntsiz)
g.ax.set_title('lr={}, lr_group={}'.format(lr, lr_group), fontsize=fntsiz)

plt.tight_layout()
# if saveplots:
#     figname = os.path.join(figdir,
#                             'gscores_violin')
#     if anneal_lr_group:
#         figname = figname + "_ann_lr_group"
#     # plt.savefig(figname, dpi=100)
#     plt.savefig(figname + '.pdf')
plt.show()


# swarm / univariate scatter
# g = sns.swarmplot(size=3, data=df_gscore)
# g.set_ylim(-.5, 1.65)
# if saveplots:
#     figname = os.path.join(figdir,
#                            'gscores_swarm.pdf')
#     # plt.savefig(figname, dpi=100)
#     plt.savefig(figname)
# plt.show()


# %% plot actmaps and xcorrs


# saveplots = False

# k
# [0.08, 0.1, 0.13, 0.26, 0.28]
# - 0.26 is 3, making a line, can remove
k = 0.1

# - set lr's for now
# [.001, .0025, .005, .0075],
# [.85, 1]]

# ok here it makes a different- slow lr's look worse
# - 0.001 worst. .0025 might be ok, but .005/.075 look good.

lr = params[1][3]
lr_group = params[2][1]

# load
fn = (
    os.path.join(wd, 'spatial_simple_ann_{:d}units_k{:.2f}_'
                  'startlr{:.4f}_grouplr{:.3f}_{:d}ktrls_'
                  '{:d}sims.pkl'.format(
                      n_units, k, lr, lr_group, n_trials//1000, n_sims))
    )
f = torch.load(fn)

# load actmap
act_maps = f['act_map']


# act_map and autocorrelogram
isim = 0

# # check which have gd gscores
# # np.nonzero(df_gscore[c].values>.5)

_, _, _, _, sac = _compute_grid_scores(act_maps[isim])


fig, ax = plt.subplots(1, 2)
ax[0].imshow(act_maps[isim])
ax[0].set_title('k = {}'.format(k))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(sac)
ax[1].set_title('g = {}'.format(np.around(df_gscore[k][isim], decimals=3)))
ax[1].set_xticks([])
ax[1].set_yticks([])
# if saveplots:
#     figname = (
#         os.path.join(figdir, 'actmaps/spatial_actmap_xcorr_annlrgroup_c{}_'
#                       '{}units_k{}_startlr{}_startgrouplr{}_thresh.7_{}trls'
#                       '_sim{}.pdf'.format(
#                           c, n_units, k, orig_lr,
#                           params['lr_clusters_group'][0], n_trials, isim))
#     )

#     plt.savefig(figname)
# plt.show()


# %% demo for figure

n_dims = 2
n_epochs = 1
n_trials = 50000
attn_type = 'dimensional_local'

# 1 / k gives slightly less than n clusters
# - e.g. if k=.28, that's > 1/4 of the units. if update k each time, there will
# be less than 4 clusters at the end - 4 will always be unstable (pulled apart)

# - did .08, .18, .28 with lr_group=0
# TODO -  get good examples for .08, .18 with lr_group > 0.

# params = [[.08, .1, .13, .18, .28],
#           [.0075, .01],
#           [.6, .8, 1.]]

p = [0.18, 0.01, 1.]

n_units = 1000

# generate path
path = generate_path(n_trials, n_dims)

k = p[0]

# annealed lr
n_trials_tmp = 500000  # to mimic main session
orig_lr = p[1]
ann_c = (1/n_trials)/n_trials_tmp
ann_decay = ann_c * (n_trials_tmp * 100)  # 100
lr = [orig_lr / (1 + (ann_decay * itrial))
      for itrial in range(n_trials_tmp)]

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

train_unsupervised_k(model, path, n_epochs)

# grid score
n_trials_test = int(n_trials * .5)  # .25+
path_test = torch.tensor(
    np.around(np.random.rand(n_trials_test, n_dims), decimals=3),
    dtype=torch.float32)

# get act
# - gauss act - define here since the units_pos don't change anymore
cov = torch.cholesky(torch.eye(2) * .01)  # cov=1
mvn1 = torch.distributions.MultivariateNormal(model.units_pos.detach(),
                                              scale_tril=cov)
nbins = 40
act_test = []
for itrial in range(n_trials_test):

    # gauss act
    act = torch.exp(mvn1.log_prob(path_test[itrial].detach()))

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


# %%  plot

saveplots = False

results = torch.stack(model.units_pos_trace, dim=0)

# group
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.scatter(results[-1, :, 0], results[-1, :, 1])
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.show()

# over time
plot_trials = torch.tensor(torch.linspace(0, n_trials, 50), dtype=torch.long)
plot_trials = torch.arange(0, 2000, 100, dtype=torch.long)

for i in plot_trials[0:-1]:  # range(20):  #

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    # stimulus pos on trial i
    # if np.mod(i, 2) == 0:
    # m = '$S$'  # stim
    # c = 'black'
    # ax.scatter(path[i, 0], path[i, 1],
    #            c=c, marker=m, s=1000, linewidth=.05, zorder=3)
    # else:
    #     m = '$c$'  # second update
    #     c = 'black' # np.array([[.5, .5, .5],])
    #     ax.scatter(results[i, :, 0].mean(), results[i, :, 1].mean(),c=c,
    #                marker=m, edgecolor='black', s=1000, linewidth=.05, zorder=3)

    # unit pos on trial i
    ax.scatter(results[i, :, 0], results[i, :, 1],
               s=350, edgecolors='black', linewidth=.5, zorder=2, alpha=.75)

    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    if saveplots:
        figname = (
            os.path.join(figdir, 'spatial_demos/spatial_unitspos_ann_{}units'
                         '_k{}_startlr{}_grouplr{}_{}ktrls_trl{}.pdf'.format(
                             n_units, k, orig_lr, lr_group, n_trials//1000, i))
        )
        plt.savefig(figname)

    plt.pause(.25)

# plot final trial
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.scatter(results[-1, :, 0], results[-1, :, 1],
           s=350, edgecolors='black', linewidth=.5, zorder=2, alpha=.75)
ax.set_xlim([-.05, 1.05])
ax.set_ylim([-.05, 1.05])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')
if saveplots:
    figname = (
        os.path.join(figdir, 'spatial_demos/spatial_unitspos_ann_{}units_k{}_'
                     'startlr{}_grouplr{}_{}ktrls_trl{}.pdf'.format(
                         n_units, k, orig_lr, lr_group, n_trials//1000,
                         n_trials))
    )
    plt.savefig(figname)

# actmap and xcorr
fig, ax = plt.subplots(1, 2)
ax[0].imshow(act_map_norm)
ax[0].set_title('k = {}'.format(k))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(sac)
ax[1].set_title('g = {}'.format(np.around(score_60_, decimals=3)))
ax[1].set_xticks([])
ax[1].set_yticks([])
if saveplots:
    figname = (
        os.path.join(figdir, 'spatial_demos/spatial_actmap_xcorr_ann_{}units_k{}_'
                     'startlr{}_grouplr{}_{}ktrls.pdf'.format(
                          n_units, k, orig_lr, lr_group, n_trials//1000))
    )

    plt.savefig(figname)
plt.show()

# # plot random scatter for positions before training
# mrksiz = 350
# fig = plt.figure(dpi=200)
# ax = fig.add_subplot(111)
# ax.scatter(model.units_pos[:, 0], model.units_pos[:, 1],
#            s=mrksiz, edgecolors='black', linewidth=.5, zorder=2, alpha=.75)
# ax.set_xlim([-.05, 1.05])
# ax.set_ylim([-.05, 1.05])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_aspect('equal', adjustable='box')
# if saveplots:
#     figname = (
#         os.path.join(figdir, 'spatial_demos/spatial_randomscatter_{}units_'
#                      'mrksiz{}.pdf'.format(n_units, mrksiz))
#     )
#     plt.savefig(figname)

# %% make gifs

savegif = True

# set params
orig_lr = .01
lr_group = 1.

n_units = 1000
n_trials = 50000

images = []

# pretraining
mrksiz = 350
fn = os.path.join(figdir, 'spatial_demos/spatial_randomscatter_{}units_'
                  'mrksiz{}.png'.format(n_units, mrksiz))

images.append(imageio.imread(fn))

# trials from plot_trials
for i in plot_trials[0:-1]:
    fn = os.path.join(figdir, 'spatial_demos/spatial_unitspos_ann_{}units_k{}_'
                      'startlr{}_grouplr{}_{}ktrls_trl{}.png'.format(
                            n_units, k, orig_lr, lr_group, n_trials//1000,
                            i))
    images.append(imageio.imread(fn))

# final trial
fn = os.path.join(figdir, 'spatial_demos/spatial_unitspos_ann_{}units_k{}_'
                  'startlr{}_grouplr{}_{}ktrls_trl{}.png'.format(
                        n_units, k, orig_lr, lr_group, n_trials//1000,
                        n_trials))
images.append(imageio.imread(fn))

if savegif:
    imageio.mimsave(
        os.path.join(figdir, 'spatial_demos/spatial_unitspos_ann_{}units_k{}_'
                     'startlr{}_grouplr{}_{}ktrls.gif'.format(
                         n_units, k, orig_lr, lr_group, n_trials//1000)),
        images, duration=.4)
