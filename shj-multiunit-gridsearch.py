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

sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
# unix
sys.path.append('/home/rm05/Documents/GitHub/multiunit-cluster')

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
maindir = '/home/rm05/Documents/'

figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-shj-gridsearch')


def negloglik(model_pr, beh_seq):
    return -np.sum(stats.norm.logpdf(beh_seq, loc=model_pr))


# define model to run
# - set up model, run through each shj problem, compute nll
def run_shj_muc(start_params, sim_info, six_problems, beh_seq):
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
                                 fit_params=True, start_params=start_params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, 16, shuffle=True, shuffle_seed=seeds[i],
            shj_order=True)

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

n_units = 500
k = .05
sim_info = {
    'n_units': n_units,
    'attn_type': 'dimensional_local',
    'k': k,
    'niter': 50  # niter
    }

lr_scale = (n_units * k) / 1

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
# ranges = ([torch.arange(1., 1.1, .1),
#           torch.arange(1., 1.1, .1),
#           torch.arange(.2, .4, .1),
#           torch.arange(.0075, .01, .0025) / lr_scale,
#           torch.arange(.075, .125, .025),
#           torch.arange(.12, .13, .01)])

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
ranges = ([torch.arange(.8, 2.1, .2),
          torch.arange(1., 19., 2),
          torch.arange(.005, .5, .05),
          torch.arange(.005, .5, .05) / lr_scale,
          torch.arange(.005, .5, .05),
          torch.arange(.1, .9, .2),  # fewer: 4 vals
          torch.tensor([.05, .1])]
          )

# try more 'ideally':
# - 1458000
# ranges = ([torch.arange(.8, 2.6, .2),
#           torch.arange(1., 19., 2),
#           torch.arange(.005, .5, .05),
#           torch.arange(.005, .5, .05) / lr_scale,
#           torch.arange(.005, .5, .05),
#           torch.arange(.1, 1., .1),
#           torch.tensor([.05, .1])]
#           )

# trying for nbanks...
# ranges = ([torch.arange(.8, 2.1, .2),
#           torch.arange(1., 19., 2),
#           torch.arange(.005, .5, .05),
#           torch.arange(.005, .5, .05) / lr_scale,
#           torch.arange(.005, .5, .05),
#           torch.arange(.1, .9, .15),
#           torch.arange(.8, 2.1, .2),
#           torch.arange(1., 19., 2),
#           torch.arange(.005, .5, .05),
#           torch.arange(.005, .5, .05) / lr_scale,
#           torch.arange(.005, .5, .05),
#           torch.arange(.1, .9, .15)]
#           )



# set up and save nll, pt, and fit_params
param_sets = torch.tensor(list(it.product(*ranges)))

# set up which subset of param_sets to run on a given run
# param_sets_curr = param_sets  # all

# for current sets, len(params_sets)=378000
# - divide by 500 sets = 756 per set
# - (756*1.875)/60=23.625 hours
# - divide by 250 sets = 1512 per set
# - (1512*1.875)/60=47.25 hours; 2 days
# note: I can run 7-8 at once with 8 cores

# test - should be 45-47 hours per set.
# - run 8 sets - result: takes 3.29 days - 79 hours
# - tested w 11 param sets, 58kb. 58 * 1512=87.696 mb per set. sounds right

# for set 1, set_n=1. do sims for: sets[set_n]:sets[set_n+1]
sets = torch.arange(0, len(param_sets)+1, 1512)
param_sets_curr = param_sets[sets[iset]:sets[iset+1]]

# testing speed
param_sets_curr = param_sets_curr[0:5]

# use list, so can combine later
pt_all = [[] for i in range(len(param_sets_curr))]
nlls = [[] for i in range(len(param_sets_curr))]
rec_all = [[] for i in range(len(param_sets_curr))]
seeds_all = [[] for i in range(len(param_sets_curr))]

# fname to save to
fn = os.path.join(datadir,
                  'shj_gsearch_{}units_k{}_set{}.pkl'.format(n_units, k, iset))

# grid search
t0 = time.time()
for i, fit_params in enumerate(param_sets_curr):

    print('Running param set {}/{} in set {}'.format(
        i+1, len(param_sets_curr), iset))

    nlls[i], pt_all[i], rec_all[i], seeds_all[i] = run_shj_muc(
        fit_params, sim_info, six_problems, beh_seq)

    # save at certain points
    if (np.mod(i, 100) == 0) | (i == len(param_sets_curr)-1):
        shj_gs_res = [nlls, pt_all, rec_all, seeds_all]
        open_file = open(fn, "wb")
        pickle.dump(shj_gs_res, open_file)
        open_file.close()

        # print time elapsed till now
        t1 = time.time()
        print(t1-t0)

t1 = time.time()
print(t1-t0)

# # to load pickled list
# open_file = open(fn, "rb")
# loaded_list = pickle.load(open_file)
# open_file.close()

"""
2.16-2.25s for 1 param value, 1 iter (6) - i.e. 1 run of 1 set of param vals
142.9s / 2.38 mins - 2 param values each, 1 iter (64)
(on love06, v similar
2.28-2.36s for 1 param value param (6), 1 iter (6)
144.95s / 2.41s for 2 values per param (6), 1 iter (64))

- this is 1 values per param and 1 sim. prob run 50 iters: 2.25*50=112.5s
112.5/60 = 1.875 mins for 1 set of param values
- just ran 1 set of params: 108.35s or 1.8 mins

average 20 steps per param
- for 6 params, there are 64,000,000 sims to run
- 1.875*64000000 = 120000000 mins = 2000000 hours = 83333.33 days = 228.3 yrs
average 10 steps, 6 params, 1,000,000 to run
- 1.875*1000000 = 1875000 mins = 31250 hours = 1302 days = 3.567 years

ok, assume above is 1 CPU, and I have n CPUs
- n = 8, 1302/8=162.75 days
- n = 16, 81.375 days
- n = 60, 21.7 days
--> maybe this is OK, on a super computer it'll be faster. 2-3 weeks

--
on unix cluster

- on login node, 0.4% of memory, 256*0.004=1.024GB. 1 sets of 50 takes 200.5s. actually slow. 


--
# intuition
c - 0.5-2 in 0.1 steps; 16 values
phi - 1-20 in 0.5 steps; 40
lr_attn - 0.005-0.5 in .015 steps; 34
lr_nn - 0.005-0.5 in .05 steps; 11
lr_clusters - .005-.5 in .05 steps; 11
lr_clusters_group - .1 to .9 in .1 steps; 9


# try to make it 10 values per param or less
c - 0.8-2 in 0.2 steps; 7
phi - 1-19 in 2 steps; 10
lr_attn - 0.005-0.5 in .05 steps; 11
lr_nn - 0.005-0.5 in .05 steps; 11
lr_clusters - .005-.5 in .05 steps; 11
lr_clusters_group - .1 to .9 in .15 steps; 6

--> this is about half of 10 values; 440980 param combinations
- 1.875*440980=826837.5/60=13780.625/24=574.19 days=2.166 years
- 574.19/25 cores = 22.9 days
- 574.19/50 cores = 11 days
- 574.19/16 = 35 days
- 574.19/8 = 71 days

with 10 values for the lr's, 378000 param combis
- 1.875*378000 = 492.1875 days = 1.348 years
- 492.1875/50 = 9.84 days
- 492.1875/16 = 30 days
- 492.1875/8 = 61.5 days


TODO:
- figure out what parameters ranges and stepsizes for each param are sensible
- split into manageable chunks - maybe split so can run some on love06/mac
then run large chunks on super computers

- remember to save periodically (outside of func, in loop)


"""

ranges = ([torch.arange(7),
          torch.arange(10),
          torch.arange(11),
          torch.arange(11),
          torch.arange(11),
          torch.arange(6)])

len(list(it.product(*ranges)))
