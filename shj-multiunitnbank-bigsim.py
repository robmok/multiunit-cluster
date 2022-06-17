#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:13:23 2022

Short version of shj-multiunitnbank.py - to run SHJ with many units and save results

@author: robert.mok
"""

import os
import sys
import torch
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

from MultiUnitClusterNBanks import (MultiUnitClusterNBanks, train)

figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

# %% SHJ

saveresults = True

set_seeds = True

n_banks = 2

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
pt_all = torch.zeros([niter, 6, n_banks+1, n_epochs])

# model details
attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
n_units = 34000000 // n_banks  # 2000
loss_type = 'cross_entropy'
k = 0.0005 / n_banks
lr_scale = (n_units * k)

rec_all =[[] for i in range(6)]
nrec_all = torch.zeros([niter, 6, n_banks])

# fc1 ws - list since diff ntrials (since it appends when recruit & upd)
w_trace = [[] for i in range(6)]
act_trace = [[] for i in range(6)]
attn_trace = [[] for i in range(6)]

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
        n_dims = inputs.shape[1]

        # gridsearch + fgsearch
        # tensor([[6.0000e-01, 1.0000e+00, 1.5500e+00, 7.0000e-01, 3.0000e-01, 8.0000e-01,
#                  1.7000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
        params = {
            'r': 1,
            'c': [.6, 1.7],
            'p': 1,
            'phi': [1., 2.25],
            'beta': 1,
            'lr_attn': [1.55, .001],
            'lr_nn': [.7/lr_scale, .01/lr_scale],
            'lr_clusters': [.3, .3],
            'lr_clusters_group': [.8, .5],
            'k': k
            }

        # gsearch + finegsearch 2022
        # tensor([[7.0000e-01, 8.7500e-01, 1.3000e+00, 8.0000e-01, 3.0000e-01, 5.0000e-01,
        #          1.7000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
        params = {
            'r': 1,
            'c': [.7, 1.7],
            'p': 1,
            'phi': [.875, 2.25],
            'beta': 1,
            'lr_attn': [1.3, .001],
            'lr_nn': [.8/lr_scale, .01/lr_scale],
            'lr_clusters': [.3, .3],
            'lr_clusters_group': [.5, .5],
            'k': k
            }

        # v2
        # tensor([[5.0000e-01, 1.2500e+00, 8.0000e-01, 7.0000e-01, 3.0000e-01, 7.0000e-01,
        #          1.8000e+00, 2.5000e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
        params = {
            'r': 1,
            'c': [.5, 1.8],
            'p': 1,
            'phi': [1.25, 2.5],
            'beta': 1,
            'lr_attn': [.8, .001],
            'lr_nn': [.7/lr_scale, .01/lr_scale],
            'lr_clusters': [.3, .3],
            'lr_clusters_group': [.7, .5],
            'k': k
            }

        model = MultiUnitClusterNBanks(n_units, n_dims, n_banks, attn_type, k,
                                       params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle_seed=seeds[i],
            shj_order=False)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        # don't save for big sim
        # w_trace[problem].append(torch.stack(model.fc1_w_trace))
        # act_trace[problem].append(torch.stack(model.fc1_act_trace))
        # attn_trace[problem].append(torch.stack(model.attn_trace))

        # save n clusters recruited
        rec_all[problem].append(model.recruit_units_trl)  # saves the seq
        # nclus recruited
        nrec_all[i, problem] = torch.tensor([len(model.recruit_units_trl[0]),
                                             len(model.recruit_units_trl[1])])

        # print(model.recruit_units_trl)

    # save variables - EDITED to save after each iter
    # - pt_all, nrec_all
    if saveresults:
        fn = os.path.join(datadir,
                          'shj_nbanks_results_pt_nrec_k{}_{}units.pkl'.format(
                              k, n_units))
        
        shj_res = [pt_all, nrec_all]  # seeds_all
        open_file = open(fn, "wb")
        pickle.dump(shj_res, open_file)
        open_file.close()
