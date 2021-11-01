#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:03:23 2021

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
    # set threads to 1 - can't do this on mac for some reason...
    torch.set_num_threads(1)

from MultiUnitClusterNBanks import (MultiUnitClusterNBanks, train)

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

    n_banks = 2

    nll_all = torch.zeros(6)
    pt_all = torch.zeros([sim_info['niter'], 6, n_banks+1, 16])
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
        model = MultiUnitClusterNBanks(sim_info['n_units'], n_dims, n_banks,
                                       sim_info['attn_type'],
                                       sim_info['k'],
                                       params=None,
                                       fit_params=True,
                                       start_params=start_params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, 16, shuffle_seed=seeds[i], shj_order=True)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()
        rec_all[problem].append(model.recruit_units_trl)

    for problem in range(6):  # TODO - check if this computing correctly
        nll_all[problem] = negloglik(pt_all[:, problem, 0].mean(axis=0),  # 0 indexing the output of the nbanks
                                     beh_seq[:, problem])

    return nll_all.sum(), torch.tensor(np.nanmean(pt_all, axis=0)), rec_all, seeds


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
    'niter': 25  # niter
    }

lr_scale = (n_units * k) / 1

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group x 2

# n-banks model - run something bigger, like 10k units, k=.01
# - with ~ 4 values per param, 5,308,416 param sets, 5.308m
# ranges = ([torch.arange(.1, 1.1, .3),
#           torch.arange(1., 15., 6),
#           torch.arange(.05, 1., .25),
#           torch.arange(.05, 1., .25) / lr_scale,
#           torch.arange(.05, 1., .25),
#           torch.arange(.2, 1., .3),

#           torch.arange(1.8, 3., .3),
#           torch.arange(1., 15., 6),
#           torch.arange(.05, 1., .25),
#           torch.arange(.05, 1., .25) / lr_scale,
#           torch.arange(.05, 1., .25),
#           torch.arange(.2, 1., .3)])

# 5308416/350000 - 15.16x of orig params
# orig took ~ 2 days for 700 sets. max is 7 days per job
# - (2200psets*.0656)/24=6.013 days
# - (2250psets*.0656)/24=6.15 days
# - (2400psets*.0656)/24=6.56 days
# - (2500*.0656)/24=6.833 days
# 5308416/2200 = 2412.916 sets to run
# - currently running 450 sets/cores at a time, ~2 days
# - 2412.916/450 = 5.362 - run 2200 psets 5.36 times
# - 2200 sets takes 6 days. so 6*5.36=32.16 days

# # cut lrs, +1 more phi 746496 psets
# ranges = ([torch.arange(.1, 1.1, .3),
#           torch.arange(1., 15., 4),
#           torch.arange(.05, 1., .4),
#           torch.arange(.05, 1., .4) / lr_scale,
#           torch.arange(.05, 1., .4),
#           torch.arange(.4, 1., .4),

#           torch.arange(2.1, 3.1, .3),
#           torch.arange(1., 15., 4),
#           torch.arange(.05, 1., .4),
#           torch.arange(.05, 1., .4) / lr_scale,
#           torch.arange(.05, 1., .4),
#           torch.arange(.4, 1., .4)]
#           )

# 746496/2200=339.31 sets to run - could run all in 1 go < 7 days
# 746496/1700= 439.115 # - even better, quicker

# or could do sth midway between above, run 2 weeks. aim for ~3.15m psets
# - more attn and nn lr's, same lr_clus
# - 2,359,296 psets
ranges = ([torch.arange(.1, 1.1, .3),
          torch.arange(1., 15., 4),
          torch.arange(.05, 1., .25),
          torch.arange(.05, 1., .25) / lr_scale,
          torch.arange(.05, 1., .35),
          torch.arange(.4, 1., .4),

          torch.arange(2.1, 3.1, .3),
          torch.arange(1., 15., 4),
          torch.arange(.05, 1., .25),
          torch.arange(.05, 1., .25) / lr_scale,
          torch.arange(.05, 1., .35),
          torch.arange(.4, 1., .4)]
          )
# # 2359296/2200sets=1072.407
# # 1072.407/450=2.383
# # 6*2.383=14.298 days - 2 weeks

# RECALCULATE - based on slower to run nbanks

# prob no need to test all params.
# - c is key - 5 values each
# - phi can be lower, esp since 2 models now - but might be useful. .5 steps
# - lr_clus and lr_group can keep same
# - attn no need to go to 1. 2nd bank can be lower - bt should i enforce ths?

# # 360000
# ranges = ([torch.arange(.2, 1.2, .2),
#           torch.arange(1., 4., .5),
#           torch.arange(.01, 1., .25),
#           torch.arange(.01, .7, .15) / lr_scale,
#           torch.tensor([.3]),
#           torch.tensor([.9]),

#           torch.arange(2., 3., .2),
#           torch.arange(1., 4., .5),
#           torch.arange(.01, 1., .25),
#           torch.arange(.01, .7, .15) / lr_scale,
#           torch.tensor([.3]),
#           torch.tensor([.9])]
#           )

# new calc -
# finalized (for now) single bank gsearch gd params (high attn)
# tensor([[0.4000, 7.0000, 2.7500, 0.0500, 0.4500, 0.9000]])
# tensor([[0.2000, 8.0000, 3.7500, 0.1450, 0.5500, 0.9000]])
# tensor([[0.2000, 8.0000, 3.2500, 0.1450, 0.6500, 0.7000]])
# - main update is lr_clus maybe need 2 vals (.3, .5)?, and group - could keep to .8...

# tested nbanks some ok gd patterns
# - c/phi values pretty sensitive, need small-ish steps
# - attn bank 2 always small..
# - nn bank 1 needs large range, bank 2 small
# params = {
#     # 'c': [.2, 2.2],
#     'c': [.25, 2.],  # .2/.25; 2./2.5
#     # 'c': [.1, 2.2],  # 2.5
#     # 'phi': [1.5, 1.5],
#     'phi': [1.8, 2.15],  # 1.9, 2.1
#     'lr_attn': [1.5, .001],  # 1.5
#     # 'lr_attn': [2., .001],  # for 3rd c/phi, meh
#     'lr_nn': [.9/lr_scale, .01/lr_scale],  # .75/.85
#     }

# 352800
# - 352800/2000 psets = 176.4 sets
# - 352800/1000 psets = 352.8 sets
# - 352800/800 = 441 sets
# - 352800/784 = 450 sets

# 358s / 5.9667 min / 0.09944 hours per set
# 0.09944*800=79.552 hours = 3.3146 days. seems doable.
# if assume cores a bit slower e.g. 8-10 mins (.133/.166 hours):
# .1333*800=106.64 hrs, 4.44 days
# .1666*800=133.28 hrs, 5.55 days
# 784 sets
# 0.09944*784=77.96/24=3.24 days
# 0.1333*784=104.51/24=4.35 days
# 0.1666*784=130.6/24=5.442 days

# test this first, see everything ok, script up analysis
# - should be able to see how off these are, and which params need to extend

# 352800 params,784 sets, 784*5=3920/60=65.4hrs=2.72 days
# 352800/400=882 sets. 882*5=4410/60/24=3.06 days
# 352800/360=980 sets. 980*5=4900/60/24=3.40277 days
ranges = ([torch.arange(.1, .7, .1),
          torch.arange(.75, 2.5, .375),
          torch.arange(.01, 3., .4),
          torch.arange(.01, 1., .15) / lr_scale,
          torch.tensor([.3]),
          torch.tensor([.8]),

          torch.arange(1.8, 2.5, .1),
          torch.arange(.75, 2.5, .375),
          torch.arange(.001, .1, .05),  # 2 vals only
          torch.arange(.01, .4, .15) / lr_scale,
          torch.tensor([.3]),
          torch.tensor([.8])]
          )

# 2nd go - STOP - redoing above as scaled twice..

# # 25 iters, 3-6 min per sim now
# # - higher phi's, nn's? not sure why plots are different to
# # just give this a try - 147456 params
# # 147456/400=368.64. 368.64*5=1843.19/60/24=1.2798 days
# ranges = ([torch.arange(.1, .5, .15),
#           torch.arange(.75, 3.5, .5),
#           torch.arange(.01, 3., .4),
#           torch.arange(.01, 1., .25) / lr_scale,
#           torch.tensor([.3]),
#           torch.tensor([.8]),

#           torch.arange(1.8, 3., .15),
#           torch.arange(.75, 6.25, .75),
#           torch.arange(.001, .1, .05),  # 2 vals only
#           torch.arange(.01, .4, .2) / lr_scale,
#           torch.tensor([.3]),
#           torch.tensor([.8])]
#           )

param_sets = torch.tensor(list(it.product(*ranges)))

# set up which subset of param_sets to run on a given run
# sets = torch.arange(0, len(param_sets), 784)  # 450 sets for nbanks
# sets = torch.arange(0, len(param_sets), 882)  # 400 sets for nbanks
sets = torch.arange(0, len(param_sets), 980)  # 360 sets for nbanks
# sets = torch.arange(0, len(param_sets), 369)  # 400 sets for nbanks2 - not run
# not a great way to add final set on
sets = torch.cat(
    [sets.unsqueeze(1), torch.ones([1, 1]) * len(param_sets)]).squeeze()
sets = torch.tensor(sets, dtype=torch.long)

param_sets_curr = param_sets[sets[iset]:sets[iset+1]]

# testing speed
# param_sets_curr = param_sets_curr[0:3]

# use list, so can combine later
pt_all = [[] for i in range(len(param_sets_curr))]
nlls = [[] for i in range(len(param_sets_curr))]
rec_all = [[] for i in range(len(param_sets_curr))]
# seeds_all = [[] for i in range(len(param_sets_curr))]

# set seeds for niters of shj problem randomised - same seqs across params
seeds = torch.arange(sim_info['niter'])*10

# fname to save to
fn = os.path.join(datadir,
                  'shj_nbanks_gsearch_k{}_{}units_set{}.pkl'.format(
                      k, n_units, iset))

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
# t0 = time.time()
for i, fit_params in enumerate(param_sets_curr[start:len(param_sets_curr)]):
    t0 = time.time()

    print('Running param set {}/{} in set {}'.format(
        i + 1 + start, len(param_sets_curr), iset))

    nlls[i + start], pt_all[i + start], rec_all[i + start], _ = run_shj_muc(
        fit_params, sim_info, six_problems, beh_seq, seeds=seeds)

    t1 = time.time()
    print(t1-t0)

    # save at certain points and at the end
    if (np.mod(i + start, 2) == 0) | (i + start == len(param_sets_curr)-1):
        shj_gs_res = [nlls, pt_all, rec_all]  # seeds_all
        open_file = open(fn, "wb")
        pickle.dump(shj_gs_res, open_file)
        open_file.close()

# t1 = time.time()
# print(t1-t0)
