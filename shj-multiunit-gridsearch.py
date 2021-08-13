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
# import matplotlib.pyplot as plt
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

# gpu if available - cpu much faster for this
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def negloglik(model_pr, beh_seq):
    return -np.sum(stats.norm.logpdf(beh_seq, loc=model_pr))


# define model to run
# - set up model, run through each shj problem, compute nll
def run_shj_muc(start_params, sim_info, six_problems, beh_seq, device):
    """
    niter: number of runs per SHJ problem with different sequences (randomised)
    """

    nll_all = torch.zeros(6)
    pt_all = torch.zeros([sim_info['niter'], 6, 16])
    rec_all = [[] for i in range(6)]

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
                                 start_params=start_params,
                                 device=device).to(device)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, 16, shuffle=True, shuffle_seed=seeds[i],
            shj_order=True, device=device)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()
        rec_all[problem].append(model.recruit_units_trl)

    for problem in range(6):
        nll_all[problem] = negloglik(pt_all[:, problem].mean(axis=0),
                                     beh_seq[:, problem])

    return nll_all.sum(), pt_all.mean(axis=0), rec_all, seeds


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
iset = 0  # 18

# for cbu-cluster
iset = int(sys.argv[-1])
# iset = 500+364 # from 500 to 864

# rest of lr's
rest_of_lrs = False

if rest_of_lrs:
    iset = iset-500  # reset when save

n_units = 2000
k = .1
sim_info = {
    'n_units': n_units,
    'attn_type': 'dimensional_local',
    'k': k,
    'niter': 50  # niter
    }

lr_scale = (n_units * k) / 1

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
# # trying
# c - 0.8-2 in 0.2 steps; 7
# phi - 1-19 in 2 steps; 10
# lr_attn - 0.005-0.455 in .05 steps; 10
# lr_nn - 0.005-0.5 in .05 steps; 10
# lr_clusters - .005-.5 in .05 steps; 10
# lr_clusters_group - .1 to .9 in .15 steps; 6

# # 378000 param sets
# ranges = ([torch.arange(.8, 2.1, .2),
#           torch.arange(1., 19., 2),
#           torch.arange(.005, .5, .05),
#           torch.arange(.005, .5, .05) / lr_scale,
#           torch.arange(.005, .5, .05),
#           torch.arange(.1, .9, .15)])

# add k, fewer lr_group
# - with 3 k values, this is double of above: 756000
# - with 2 k values (just to see how different): 504000. orig is .75 of this. will take 1.33*
# ranges = ([torch.arange(.8, 2.1, .2),
#           torch.arange(1., 19., 2),
#           torch.arange(.005, .5, .05),
#           torch.arange(.005, .5, .05) / lr_scale,
#           torch.arange(.005, .5, .05),
#           torch.arange(.1, .9, .2),  # fewer: 4 vals
#           torch.tensor([.05, .1])]
#           )

# without k - 252000
ranges = ([torch.arange(.8, 2.1, .2),
          torch.arange(1., 19., 2),
          torch.arange(.005, .5, .05),
          torch.arange(.005, .5, .05) / lr_scale,
          torch.arange(.005, .5, .05),
          torch.arange(.1, .9, .2)]  # fewer: 4 vals
          )
# add more lr's
# - if double (same nsetss - 25k), 10 more: .5 to .95. arange(.5, 1., .05)
# - could go up to .9, then 183708 sets - prob this
# - if .85, 129024
if rest_of_lrs:
    ranges = ([torch.arange(.8, 2.1, .2),
              torch.arange(1., 19., 2),
              torch.arange(.5, .95, .05),
              torch.arange(.5, .95, .05) / lr_scale,
              torch.arange(.5, .95, .05),
              torch.arange(.1, .9, .2)]  # fewer: 4 vals
              )

# # new - coarser
# ranges = ([torch.arange(.8, 2.1, .3),
#           torch.arange(1., 19., 2.5),
#           torch.arange(.005, .5, .1),
#           torch.arange(.005, .5, .1) / lr_scale,
#           torch.arange(.005, .5, .1),
#           torch.arange(.1, .9, .2)]
#           )

# set up and save nll, pt, and fit_params
param_sets = torch.tensor(list(it.product(*ranges)))

# set up which subset of param_sets to run on a given run
# param_sets_curr = param_sets  # all

# update - cluster is faster than expected
# - 160 sets in 8.5 or 11.5 hours
# - 160/8.5=18.8235 or 160/10.5=15.2381 sets per hour. Let's say 15 per hour.
# - 	8.5/160=0.0531 or 10.5/160=0.0656 hours per set

# w 378000
# - divide by 500 sets = 756 per set
# - divide by 250 sets = 1512 per set
# - divide by 125 sets = 3024 per set
# .0656*756=49.59/24=2.066, .0656*1512=99.1872/24=4.13 days (x2 since 250)
# .0656*3024=198.3744/24=8.26 days

# w 252000
# 252000/250=1008*.0656=66.1248/24 = 2.7552 days (x2 = 5.51 days)

# let's try 252000, 250 sets (1008 per set); x2 = 5.51 days
# - run k=0.05, run sbatch w 250 jobs (should run 128 then queue)
# - then run another sbatch with k=0.01 - check later if this times out since
# it'll be 7+ days. but maybe ok since it's in a queue, not wall time?

# NEW - now got lowpri - more cores:
# - orig would've taken 2.755 days
# 252000/350sets=720*.0656=47.232/24=1.96 days
# 252000/400=630*.0656=41.328/24=1.72
# 252000/450=560*.0656=36.736/24=1.53
# 252000/500=504*.0656=33.0624=1.3776

# 500 sets
# - Run 0-127 jobs (128 cores) on normal priority
# - Run 128-499 (372 cores) on lopri.

# rest of lr's - 365 param sets (arange(.5, .95, .05) - up to .9)
# 183708/365 = 503.3095*.0656=33.0171/24=1.37  days
# - just do 504, final one have about half

sets = torch.arange(0, len(param_sets)+1, 504)

# TMP
# add final set if doesn't mod - for 2nd part of lr's
if rest_of_lrs:
    sets = torch.cat(
        [sets.unsqueeze(1), torch.ones([1, 1]) * len(param_sets)]).squeeze()
    sets = torch.tensor(sets, dtype=torch.long)


param_sets_curr = param_sets[sets[iset]:sets[iset+1]]

if rest_of_lrs:
    iset = iset+500  # for saving

# testing speed
# param_sets_curr = param_sets_curr[0:1]

# use list, so can combine later
pt_all = [[] for i in range(len(param_sets_curr))]
nlls = [[] for i in range(len(param_sets_curr))]
rec_all = [[] for i in range(len(param_sets_curr))]
seeds_all = [[] for i in range(len(param_sets_curr))]

# fname to save to
fn = os.path.join(datadir,
                  'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

# grid search
t0 = time.time()
for i, fit_params in enumerate(param_sets_curr):

    print('Running param set {}/{} in set {}'.format(
        i+1, len(param_sets_curr), iset))

    nlls[i], pt_all[i], rec_all[i], seeds_all[i] = run_shj_muc(
        fit_params, sim_info, six_problems, beh_seq, device=device)

    # save at certain points
    if (np.mod(i, 100) == 0) | (i == len(param_sets_curr)-1):
        shj_gs_res = [nlls, pt_all, rec_all, seeds_all]
        open_file = open(fn, "wb")
        pickle.dump(shj_gs_res, open_file)
        open_file.close()

        # print time elapsed till now
    # t1 = time.time()
    # print(t1-t0)

t1 = time.time()
print(t1-t0)
