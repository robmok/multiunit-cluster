#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:30:33 2021

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import itertools as it
import time
from scipy import stats
import pickle

location = 'cluster'  # 'mbp' or 'cluster' (cbu cluster - unix)

if location == 'mbp':
    maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
    sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
elif location == 'cluster':
    maindir = '/imaging/duncan/users/rm05/'
    sys.path.append('/home/rm05/Documents/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-shj-gridsearch')


def negloglik(model_pr, beh_seq):
    return -np.sum(stats.norm.logpdf(beh_seq, loc=model_pr))


# define model to run
# - set up model, run through each shj problem, compute nll
def run_shj_muc(start_params, sim_info, six_problems, beh_seq,
                seeds=None):
    """
    niter: number of runs per SHJ problem with different sequences (randomised)
    """

    nll_all = torch.zeros(6)
    pt_all = torch.zeros([sim_info['niter'], 6, 16])
    rec_all = [[] for i in range(6)]

    if seeds is None:  # else, put in the seed values
        seeds = torch.randperm(sim_info['niter']*100)[:sim_info['niter']]

    # run niterations, 6 problems
    for problem, i in it.product(range(6), range(sim_info['niter'])):

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
                                 fit_params=True,
                                 start_params=start_params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, 16, shuffle=True, shuffle_seed=seeds[i],
            shj_order=True)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()
        rec_all[problem].append(model.recruit_units_trl)

    for problem in range(6):
        nll_all[problem] = negloglik(pt_all[:, problem].mean(axis=0),
                                     beh_seq[:, problem])

    return (nll_all.sum(), torch.tensor(np.nanmean(pt_all, axis=0)), rec_all,
            seeds)


# %% grid search, fit shj

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
beh_seq = shj.T


# start
iset = 0

# for cbu-cluster
iset = int(sys.argv[-1])

n_units = 2000
k = .01
sim_info = {
    'n_units': n_units,
    'attn_type': 'dimensional_local',
    'k': k,
    'niter': 50  # niter
    }

lr_scale = (n_units * k) / 1

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group

# whole range of lrs, added group lr too
# ranges = ([torch.arange(.8, 2.1, .2),
#           torch.arange(1., 19., 2),
#           torch.arange(.05, 1., .1),
#           torch.arange(.05, 1., .1) / lr_scale,
#           torch.arange(.05, 1., .1),
#           torch.arange(.1, 1., .2)]
#           )

# # edit - -2 phi (all gd params were 1!), and 2+ lower c values
# ranges = ([torch.arange(.4, 2.1, .2),
#           torch.arange(1., 15., 2),
#           torch.arange(.05, 1., .1),
#           torch.arange(.05, 1., .1) / lr_scale,
#           torch.arange(.05, 1., .1),
#           torch.arange(.1, 1., .2)]
#           )

# add 1 more c value
ranges = ([torch.arange(.2, 2.1, .2),
          torch.arange(1., 15., 2),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1) / lr_scale,
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

# set up and save nll, pt, and fit_params
param_sets = torch.tensor(list(it.product(*ranges)))

# timing
# - 160 sets in 8.5 or 11.5 hours
# - 160/8.5=18.8235 or 160/10.5=15.2381 sets per hour. Let's say 15 per hour.
# - 8.5/160=0.0531 or 10.5/160=0.0656 hours per set

# NEW - decided to test full range of lr's coarser - 315000 sets
# - 315000/500=630*.0656=41.328/24=1.72 days
# ah, but i want lopri to be ~300, so has to be less
# - 315000/450=700*.0656=45.92=1.91 days. 322 lopri sets
# - 315000/400=787.5*.0656=51.66/24=2.1525 days. 272 lopri sets, def OK.
# --> 450

# added 2 lower c's, removed 2 higher phi's

# added another lower c val
# 350000/450=777.77*.0656=51.02/24=2.125 days

param_sets = torch.tensor(list(it.product(*ranges)))

# set up which subset of param_sets to run on a given run
sets = torch.arange(0, len(param_sets)+1, 778)  # 2200 for nbanks
# not a great way to add final set on
sets = torch.cat(
    [sets.unsqueeze(1), torch.ones([1, 1]) * len(param_sets)]).squeeze()
sets = torch.tensor(sets, dtype=torch.long)

param_sets_curr = param_sets[sets[iset]:sets[iset+1]]

# testing speed
# param_sets_curr = param_sets_curr[0:1]

# use list, so can combine later
pt_all = [[] for i in range(len(param_sets_curr))]
nlls = [[] for i in range(len(param_sets_curr))]
rec_all = [[] for i in range(len(param_sets_curr))]
# seeds_all = [[] for i in range(len(param_sets_curr))]

# set seeds for niters of shj problem randomised - same seqs across params
seeds = torch.arange(sim_info['niter'])*10

# fname to save to
fn = os.path.join(datadir,
                  'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

# on mbp testing
# fn = os.path.join(datadir, 'gsearch_k0.05_1000units/'
#                'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

# if file exists, load up and rerun
if os.path.exists(fn):
    open_file = open(fn, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    # load back in
    nlls = loaded_list[0]
    pt_all = loaded_list[1]
    rec_all = loaded_list[2]

    # find where to restart from
    ind = [nlls[i] == [] for i in range(len(param_sets_curr))]
    start = torch.nonzero(torch.tensor(ind)).min()
else:
    start = 0

# grid search
t0 = time.time()
for i, fit_params in enumerate(param_sets_curr[start:len(param_sets_curr)]):

    print('Running param set {}/{} in set {}'.format(
        i + 1 + start, len(param_sets_curr), iset))

    nlls[i], pt_all[i], rec_all[i], _ = run_shj_muc(
        fit_params, sim_info, six_problems, beh_seq, seeds=seeds)

    # save at certain points and at the end
    if (np.mod(i + start, 100) == 0) | (i + start == len(param_sets_curr)-1):
        shj_gs_res = [nlls, pt_all, rec_all]  # seeds_all
        open_file = open(fn, "wb")
        pickle.dump(shj_gs_res, open_file)
        open_file.close()

        # print time elapsed till now
    # t1 = time.time()
    # print(t1-t0)

t1 = time.time()
print(t1-t0)
