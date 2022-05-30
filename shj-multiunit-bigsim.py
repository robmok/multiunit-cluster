#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:22:41 2022

Short version of shj-multiunit.py - to run SHJ with many units and save results

Plus make some 3D plots over time

@author: robert.mok
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
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

# TODO - create dirs if not yet on cluster
figdir = os.path.join(maindir, 'multiunit-cluster_figs')
datadir = os.path.join(maindir, 'muc-results')

# set n_units and k
n_units = 3400000  # 3400000 # 2000
k = 0.0005  # .05% winners

# %% SHJ 6 problems

# saveresults = True

# set_seeds = True

# six_problems = [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0],
#                  [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

#                 [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1],
#                  [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]],

#                 [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
#                  [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]],

#                 [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
#                  [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],

#                 [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
#                  [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],

#                 [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
#                  [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],

#                 ]

# niter = 25

# # set seeds for niters of shj problem randomised - same seqs across params
# if set_seeds:
#     seeds = torch.arange(1, niter+1)*10  # - what was used for gridsearch
# else:
#     seeds = torch.randperm(niter*100)[:niter]

# n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
# pt_all = torch.zeros([niter, 6, n_epochs])
# rec_all =[[] for i in range(6)]
# nrec_all = torch.zeros([niter, 6])
# w_trace = [[] for i in range(6)]
# attn_trace = [[] for i in range(6)]

# # run multiple iterations
# for i in range(niter):

#     # six problems

#     for problem in range(6):  # [0, 5]: #  np.array([4]):

#         stim = six_problems[problem]
#         stim = torch.tensor(stim, dtype=torch.float)
#         inputs = stim[:, 0:-1]
#         output = stim[:, -1].long()  # integer

#         # 16 per trial
#         inputs = inputs.repeat(2, 1)
#         output = output.repeat(2).T

#         # model details
#         attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
#         n_dims = inputs.shape[1]
#         loss_type = 'cross_entropy'

#         # scale lrs - params determined by n_units=100, k=.01. n_units*k=1
#         lr_scale = (n_units * k) / 1

#         # final - for plotting
#         # tensor([[ 0.2000, 5/11,  3.0000,  0.0750/0.3750,  0.3250,  0.7000]])
#         # - type 3 bit faster, but separation with 6 better, overall slower
#         # and i think more canonical sustain recruitments. choose this?
#         params = {
#             'r': 1,  # 1=city-block, 2=euclid
#             'c': .2,
#             'p': 1,
#             'phi': 5.,  # 5/11
#             'beta': 1.,
#             'lr_attn': 3.,  # .95,  # this scales at grad computation now
#             'lr_nn': .375/lr_scale,  # .075/0.3750
#             'lr_clusters': .325,
#             'lr_clusters_group': .7,
#             'k': k
#             }
#         # OR
#         # params = {
#         #     'r': 1,  # 1=city-block, 2=euclid
#         #     'c': .2,
#         #     'p': 1,
#         #     'phi': 11.,  # 5/11
#         #     'beta': 1.,
#         #     'lr_attn': 3.,  # .95,  # this scales at grad computation now
#         #     'lr_nn': .075/lr_scale,  # .075/0.3750
#         #     'lr_clusters': .325,
#         #     'lr_clusters_group': .7,
#         #     'k': k
#         #     }

#         model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

#         model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
#             model, inputs, output, n_epochs,  shuffle_seed=seeds[i],
#             shj_order=True)

#         pt_all[i, problem] = 1 - epoch_ptarget.detach()

#         # - don't save when doing big sim
#         # w_trace[problem].append(torch.stack(model.fc1_w_trace))
#         # attn_trace[problem].append(torch.stack(model.attn_trace))

#         # save n clusters recruited
#         rec_all[problem].append(model.recruit_units_trl)  # saves the seq
#         nrec_all[i, problem] = len(model.recruit_units_trl)  # nclus recruited

#         print(model.recruit_units_trl)

# # t1 = time.time()
# # print(t1-t0)

# # save variables
# # - pt_all, nrec_all
# if saveresults:
#     fn = os.path.join(datadir,
#                       'shj_results_pt_nrec_k{}_{}units.pkl'.format(k, n_units))
    
#     shj_res = [pt_all, nrec_all]  # seeds_all
#     open_file = open(fn, "wb")
#     pickle.dump(shj_res, open_file)
#     open_file.close()

# %%  SHJ 3D plots

niter = 5  # do a few so can select a good one for figures (this is the seed)

saveplots = True  # 3d plots

plot3d = True
plot_seq = 'trls'  # 'epoch'=plot whole epoch in sections. 'trls'=1st ntrials

seeds = torch.arange(1, niter+1)*10


# matplotlib first 6 default colours
col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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

# problem = 4

for isim in range(niter):

    for problem in range(6):

        stim = six_problems[problem]
        stim = torch.tensor(stim, dtype=torch.float)
        inputs = stim[:, 0:-1]
        output = stim[:, -1].long()  # integer
        
        # 16 per trial
        inputs = inputs.repeat(2, 1)
        output = output.repeat(2).T

        # model details
        attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
        # n_units = 50000
        n_dims = inputs.shape[1]
        loss_type = 'cross_entropy'
        # k = .005  # .05
        
        # trials, etc.
        n_epochs = 16
        
        # new local attn - scaling lr
        lr_scale = (n_units * k) / 1
        
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .2,
        #     'p': 1,
        #     'phi': 7.,  # 5/11
        #     'beta': 1.,
        #     'lr_nn': .175/lr_scale,  # .075/0.3750
        #     'lr_attn': .4,
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .25,
        #     'k': k
        #     }
        
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .2,
            'p': 1,
            'phi': 5.,  # 5/11
            'beta': 1.,
            'lr_attn': .4,  # /(n_units*k), # 3., # maybe should scale here..!
            'lr_nn': .375/lr_scale,  # .075/0.3750
            'lr_clusters': .325,
            'lr_clusters_group': .7,
            'k': k
            }

        # gridsearch final
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .2,
            'p': 1,
            'phi': 14.,
            'beta': 1.,
            'lr_attn': .275,  # /(n_units*k), # 3., # maybe should scale here..!
            'lr_nn': .05/lr_scale,  # .075/0.3750
            'lr_clusters': .35,
            'lr_clusters_group': .9,
            'k': k
            }

        # lesioning
        lesions = None  # if no lesions
        
        # noise - mean and sd of noise to be added
        # - with update noise, higher lr_group helps save a lot even with few k units.
        # actually didn't add update2 noise though, test again
        noise = None
        noise = {'update1': [0, .05],  # 0, .15 1unit position updates 1 & 2
                  'update2': [0, .0],  # no noise here also makes sense - since there is noise in 1 and you get all that info.
                  'recruit': [0., .1],  # .1 recruitment position placement
                  'act': [.5, .1]}  # unit activations (non-negative)
        
        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)
        
        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle_seed=seeds[isim],
            lesions=lesions,
            noise=noise, shj_order=False)
        
        
        print(model.recruit_units_trl)
        # print(len(model.recruit_units_trl))
        print(epoch_ptarget)
        
        # # pr target
        # plt.plot(1 - epoch_ptarget.detach())
        # plt.ylim([0, .5])
        # plt.show()
        
        # # attention weights
        # plt.plot(torch.stack(model.attn_trace, dim=0))
        # # figname = os.path.join(figdir,
        # #                        'SHJ_attn_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
        # #                            problem, k, n_units, params['lr_attn'], n_epochs))
        # # plt.savefig(figname)
        # plt.show()
        
        # plot 3d - unit positions over time
        results = torch.stack(
            model.units_pos_bothupd_trace, dim=0)[:, model.active_units]
        
        if plot_seq == 'epoch':  # plot from start to end in n sections
            n_ims = 9 # 9 = 1 im per 2 blocks (16 trials * 2 (2nd update))
            plot_trials = torch.tensor(
                torch.linspace(0, len(inputs) * n_epochs, n_ims), dtype=torch.long)
        
            # problem=2/3, 6 clus needed this
            # n_ims = 18 # full
            # plot_trials = torch.tensor(
            #     torch.linspace(0, len(inputs) * n_epochs * 2, n_ims), dtype=torch.long)
            # plot_trials[-1] = plot_trials[-1]-1  # last trial
        
        elif plot_seq == 'trls':  # plot first n trials
            plot_n_trials = 50  # 80
            plot_trials = torch.arange(plot_n_trials)
            # add a later set of trials
            plot_trials = torch.cat((plot_trials, torch.tensor([len(results)-2,
                                                                len(results)-1])), 0)
        
        # 3d
        # make dir for trial-by-trial images
        if noise and saveplots:
            dn = ('dupd_shj3d_{}_type{}_{}units_k{}_lr{}_grouplr{}_c{}_phi{}_attn{}_'
                  'nn{}_upd1noise{}_recnoise{}_sim{}'.format(
                      plot_seq, problem+1, n_units, k, params['lr_clusters'],
                      params['lr_clusters_group'], params['c'], params['phi'],
                      params['lr_attn'], params['lr_nn'], noise['update1'][1],
                      noise['recruit'][1], isim)
                  )
        
            if not os.path.exists(os.path.join(figdir, dn)):
                os.makedirs(os.path.join(figdir, dn))
        
        if plot3d:
            lims = (0, 1)
            # lims = (-.05, 1.05)
            for i in plot_trials:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=150)
                ax.scatter(results[i, :, 0],
                           results[i, :, 1],
                           results[i, :, 2], c=col[problem])
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_zlim(lims)
        
                # keep grid lines, remove labels
                # # labels = ['', '', '', '', '', '']
                labels = [0, '', '', '', '', 1]
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                ax.set_zticklabels(labels)
        
                # remove grey color - white
                ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
                ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
                ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        
                # save
                if saveplots:
                    figname = os.path.join(figdir, dn, 'trial{}_sim{}'.format(i, isim))
                    print(figname)
                    plt.savefig(figname + '.png')
                    plt.savefig(figname + '.pdf')

                if not location == 'cluster':  # no need to pause if cluster
                    plt.pause(.2)

        # clear some RAM
        del results, plot_trials