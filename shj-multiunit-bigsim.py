#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:22:41 2022

Short version of shj-multiunit.py - to run SHJ with many units and save results

@author: robert.mok
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools as it
import time
import pickle

location = 'cluster'  # 'mbp' or 'cluster' (cbu cluster - unix)

if location == 'mbp':
    maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
    sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
elif location == 'cluster':
    maindir = '/imaging/duncan/users/rm05/'
    sys.path.append('/home/rm05/Documents/multiunit-cluster')
    # set threads to 1 - can't do this on mac for some reason...
    # torch.set_num_threads(1)

from MultiUnitCluster import (MultiUnitCluster, train)

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
# figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

# %% SHJ 6 problems

saveresults = True

set_seeds = True

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

niter = 25

# set seeds for niters of shj problem randomised - same seqs across params
if set_seeds:
    seeds = torch.arange(1, niter+1)*10  # - what was used for gridsearch
else:
    seeds = torch.randperm(niter*100)[:niter]

n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
pt_all = torch.zeros([niter, 6, n_epochs])
rec_all =[[] for i in range(6)]
nrec_all = torch.zeros([niter, 6])
w_trace = [[] for i in range(6)]
attn_trace = [[] for i in range(6)]

# t0 = time.time()

# run multiple iterations
for i in range(niter):

    # six problems

    for problem in range(6):  # [0, 5]: #  np.array([4]):

        stim = six_problems[problem]
        stim = torch.tensor(stim, dtype=torch.float)
        inputs = stim[:, 0:-1]
        output = stim[:, -1].long()  # integer

        # 16 per trial
        inputs = inputs.repeat(2, 1)
        output = output.repeat(2).T

        # model details
        attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
        n_units = 3400000 # 2000
        n_dims = inputs.shape[1]
        loss_type = 'cross_entropy'
        k = .01  # top k%. so .05 = top 5%

        # scale lrs - params determined by n_units=100, k=.01. n_units*k=1
        lr_scale = (n_units * k) / 1

        # final - for plotting
        # tensor([[ 0.2000, 5/11,  3.0000,  0.0750/0.3750,  0.3250,  0.7000]])
        # - type 3 bit faster, but separation with 6 better, overall slower
        # and i think more canonical sustain recruitments. choose this?
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .2,
            'p': 1,
            'phi': 5.,  # 5/11
            'beta': 1.,
            'lr_attn': 3.,  # .95,  # this scales at grad computation now
            'lr_nn': .375/lr_scale,  # .075/0.3750
            'lr_clusters': .325,
            'lr_clusters_group': .7,
            'k': k
            }
        # OR
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .2,
        #     'p': 1,
        #     'phi': 11.,  # 5/11
        #     'beta': 1.,
        #     'lr_attn': 3.,  # .95,  # this scales at grad computation now
        #     'lr_nn': .075/lr_scale,  # .075/0.3750
        #     'lr_clusters': .325,
        #     'lr_clusters_group': .7,
        #     'k': k
        #     }

        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs,  shuffle_seed=seeds[i],
            shj_order=True)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        # - don't save when doing big sim
        # w_trace[problem].append(torch.stack(model.fc1_w_trace))
        # attn_trace[problem].append(torch.stack(model.attn_trace))

        # save n clusters recruited
        rec_all[problem].append(model.recruit_units_trl)  # saves the seq
        nrec_all[i, problem] = len(model.recruit_units_trl)  # nclus recruited

        print(model.recruit_units_trl)

# t1 = time.time()
# print(t1-t0)

# save variables
# - pt_all, nrec_all
if saveresults:
    fn = os.path.join(datadir,
                      'shj_results_pt_nrec_k{}_{}units.pkl'.format(k, n_units))
    
    shj_res = [pt_all, nrec_all]  # seeds_all
    open_file = open(fn, "wb")
    pickle.dump(shj_res, open_file)
    open_file.close()