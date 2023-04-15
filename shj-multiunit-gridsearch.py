#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:30:33 2021

Run gridsearch to fit SHJ

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

from MultiUnitCluster import (MultiUnitCluster, train)

figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-shj-gridsearch')

finegsearch = True

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
            model, inputs, output, 16, shuffle_seed=seeds[i],
            shj_order=False)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()
        rec_all[problem].append(model.recruit_units_trl)

    for problem in range(6):
        nll_all[problem] = negloglik(pt_all[:, problem].mean(axis=0),
                                     beh_seq[:, problem])

    return (nll_all.sum(), torch.tensor(np.nanmean(pt_all, axis=0)), rec_all,
            seeds)
    # return (nll_all.sum(), np.nanmean(pt_all, axis=0), rec_all,
    #         seeds)

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
if location == 'cluster':
    iset = int(sys.argv[-1])

n_units = 10000
k = .005
sim_info = {
    'n_units': n_units,
    'attn_type': 'dimensional_local',
    'k': k,
    'niter': 25  # niter
    }

lr_scale = (n_units * k) / 1

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group

# 224000 params
ranges = ([torch.arange(.2, 2.1, .2),
          torch.arange(1., 15., 2),
          # torch.arange(1., 2.76, .25),
          torch.arange(1., 3.76, .25),  # more attn
          torch.arange(.05, .76, .1) / lr_scale,
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

if finegsearch:
    # 147000 sets
    ranges = ([torch.arange(.1, .71, .1),
          torch.hstack([torch.arange(3., 9., 1), torch.arange(11., 15., 1)]),
          torch.arange(.75, 3.01, .25),
          torch.hstack([torch.arange(.025, .125, .025),
                        torch.arange(.3, .43, .025)]) / lr_scale,  # 2 more
          torch.arange(.05, .7, .1),  # 1 less
          torch.arange(.5, 1., .2)]  # 1 extra
          )

param_sets = torch.tensor(list(it.product(*ranges)))

# - 336000
sets = torch.arange(0, len(param_sets), 840) # dist, 400 sets

# finegsearch
# - 147000 - 350 sets, (420 psets)
if finegsearch:
    sets = torch.arange(0, len(param_sets), 420)

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
seeds = torch.arange(1, sim_info['niter']+1)*10

# fname to save to
fn = os.path.join(datadir,
                  'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))


if finegsearch:
    fn = os.path.join(datadir,
                      'shj_finegsearch_k{}_{}units_set{}.pkl'.format(
                          k, n_units, iset))

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

    nlls[i + start], pt_all[i + start], rec_all[i + start], _ = run_shj_muc(
        fit_params, sim_info, six_problems, beh_seq, seeds=seeds)

    # save at certain points and at the end
    if (np.mod(i + start, 50) == 0) | (i + start == len(param_sets_curr)-1):
        print('a')
        shj_gs_res = [nlls, pt_all, rec_all]  # seeds_all
        open_file = open(fn, "wb")
        pickle.dump(shj_gs_res, open_file)
        open_file.close()

        # print time elapsed till now
    # t1 = time.time()
    # print(t1-t0)

t1 = time.time()
print(t1-t0)
