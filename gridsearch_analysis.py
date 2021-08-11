#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:57:19 2021

@author: robert.mok
"""

import os
# import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import itertools as it

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

k = 0.01
n_units = 500

n_sets = 250  # gsearch split into how many sets to load in

resdir = os.path.join(maindir, 'muc-shj-gridsearch/gsearch_k{}_{}units'.format(
    k, n_units))

# get params - 252000
ranges = ([torch.arange(.8, 2.1, .2),
          torch.arange(1., 19., 2),
          torch.arange(.005, .5, .05),
          torch.arange(.005, .5, .05), #  / lr_scale,  # ignoring this here
          torch.arange(.005, .5, .05),
          torch.arange(.1, .9, .2)]
          )
param_sets = torch.tensor(list(it.product(*ranges)))

# load in
pts = []
nlls = []
recs = []
seeds = []

sets = torch.arange(n_sets)
# sets = sets[sets != 142]  # k=0.05, n_units=500, set 142 has some results but most empty - didn't finish?

for iset in sets:  # range(n_sets):
    fn = os.path.join(
        resdir,
        'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    # load - list: [nlls, pt_all, rec_all, seeds_all]
    open_file = open(fn, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    if not loaded_list[1]:
        print(iset)

    nlls.extend(loaded_list[0])
    pts.extend(loaded_list[1])
    # recs.extend(loaded_list[2])
    # seeds.extend(loaded_list[3])

pts = torch.stack(pts)
nlls = torch.stack(nlls)
# recs = torch.stack(recs)
# seeds = torch.stack(seeds)

# %% fit

# TODO: i probably should have a set of sequences to run on each of the param
# sets so they'd be the same across k conditions...
# - when i run the "big" one, set and save the seeds


# the human data from nosofsky, et al. replication
shj = (
    torch.tensor(
        [[0.211, 0.025, 0.003, 0., 0., 0., 0., 0.,
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
          0.172, 0.128, 0.139, 0.117, 0.103, 0.098, 0.106, 0.106]]).T
    )

# criteria for fit:

# quantitative:
# - compute SSE for each fit (already have nll)

# qualitiative
# - pattern - 1, 2, 3-4-5, 6 (where 3, 4, 5 can be in any order for now). all
# points have to be faster (for now - maybe do 80% points if problem?)
# -

# iparam = 0

# match threshold
# - 1 if match fully. can allow some error to be safe, eg ~.9
# - note depending on the comparison, num total is diff (so prop is diff)
match_thresh = .95

# pattern, meet criteria?
sse = torch.zeros(len(pts))
ptn_criteria_1 = torch.zeros(len(pts), dtype=torch.bool)
for iparam in range(len(pts)):

    # compute sse
    sse[iparam] = torch.sum(torch.square(
        pts[iparam].T.flatten() - shj.flatten()))

    # pattern
    ptn = pts[iparam][0] < pts[iparam][1:]  # type I fastest
    ptn_c1 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    ptn = pts[iparam][1] < pts[iparam][2:]  # type II 2nd
    ptn_c2 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    ptn = pts[iparam][5] > pts[iparam][:5]  # type VI slowest
    ptn_c3 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh

    ptn_criteria_1[iparam] = ptn_c1 & ptn_c2 & ptn_c3


# criteria 1 already reduces to <12% of the params

# sse and nll very similar, though LOWEST value differ. interesting
plt.plot(nlls[ptn_criteria_1])
plt.ylim([88, 97])
# plt.ylim([88, 89])
plt.show()
plt.plot(sse[ptn_criteria_1])
plt.ylim([0, 16.5])
# plt.ylim([0, 1])
plt.show()

ind_nll = nlls == nlls[ptn_criteria_1].min()
ind_sse = sse == sse[ptn_criteria_1].min()

# ind_nll = nlls == nlls.min()
# ind_sse = sse == sse.min()

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
print(param_sets[ind_nll])
print(param_sets[ind_sse])

# more criteria
# - maybe faster type I / slower type VI
# - types III-V need to be more similar (e.g. within some range)
# - type I, II and the III-V's need to be larger differences



# %% plot

saveplots = False

fntsiz = 15
ylims = (0., .55)

fig, ax = plt.subplots(2, 1)
ax[0].plot(shj)
ax[0].set_ylim(ylims)
ax[0].set_aspect(17)
ax[0].legend(('1', '2', '3', '4', '5', '6'), fontsize=7)
ax[1].plot(pts[ind_sse].T.squeeze())
ax[1].set_ylim(ylims)
ax[1].set_aspect(17)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'shj_gsearch_n94_subplots_{}units_k{}_1.png'.format(
                               n_units, k))
    plt.savefig(figname)
plt.show()

# best params by itself
fig, ax = plt.subplots(1, 1)
ax.plot(pts[ind_sse].T.squeeze())
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.legend(('1', '2', '3', '4', '5', '6'), fontsize=fntsiz-2)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'shj_gsearch_{}units_k{}_1.png'.format(
                               n_units, k))
    plt.savefig(figname)
plt.show()

# nosofsky '94 by itself
fig, ax = plt.subplots(1, 1)
ax.plot(shj)
ax.set_ylim(ylims)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.legend(('1', '2', '3', '4', '5', '6'), fontsize=fntsiz-2)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir, 'nosofsky94_shj.png')
    plt.savefig(figname)
plt.show()

# plot on top of each other
fig, ax = plt.subplots(1, 1)
ax.plot(shj, 'k')
ax.plot(pts[ind_sse].T.squeeze(), 'o-')
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
plt.show()

