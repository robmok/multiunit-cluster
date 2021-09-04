#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:57:19 2021

@author: robert.mok
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import itertools as it

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

k = 0.01
n_units = 2000

n_sets = 450  # 865  # 250  # gsearch split into how many sets to load in

resdir = os.path.join(maindir, 'muc-shj-gridsearch/gsearch_k{}_{}units'.format(
    k, n_units))

ranges = ([torch.arange(.4, 2.1, .2),
          torch.arange(1., 15., 2),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

param_sets = torch.tensor(list(it.product(*ranges)))

sets = torch.arange(n_sets)

# TMP
# sets = sets[(sets != 80) & (sets != 109)]  # TMP remove sets 80 and 109

# # TMP - remove some sets if incomplete - 80, 109
# sets = torch.arange(0, len(param_sets)+1, 700)
# ind = torch.ones(len(param_sets), dtype=torch.bool)
# ind[sets[80]:sets[81]] = False
# ind[sets[109]:sets[110]] = False
# param_sets = param_sets[ind]

# load in
pts = []
nlls = []
recs = []
seeds = []

for iset in sets:  # range(n_sets):
    fn = os.path.join(
        resdir,
        'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    # load - list: [nlls, pt_all, rec_all, seeds_all]
    open_file = open(fn, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    # if not loaded_list[1]:
    #     print(iset)

    if not np.any(loaded_list[1][-1]):  # check last one
        print(iset)

    nlls.extend(loaded_list[0])
    pts.extend(loaded_list[1])
    # recs.extend(loaded_list[2])
    # seeds.extend(loaded_list[3])


# pts = torch.stack(pts)
# nlls = torch.stack(nlls)
# recs = torch.stack(recs)
# seeds = torch.stack(seeds)

# after doing nan mean, these are now numpy arrays. will change later
# pts = np.stack(pts)
# nlls = np.stack(nlls)

# %% fit

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
# - compute SSE for differences

# qualitiative
# - pattern - 1, 2, 3-4-5, 6 (where 3, 4, 5 can be in any order for now). all
# points have to be faster (for now - maybe do 80% points if problem?)

# iparam = 0

# match threshold
# - 1 if match fully. can allow some error to be safe, eg ~.9
# - note depending on the comparison, num total is diff (so prop is diff)
match_thresh = .95

# criterion 1 - shj pattern (qualitative)
sse = torch.zeros(len(pts))
ptn_criteria_1 = torch.zeros(len(pts), dtype=torch.bool)

# criterion 2 (quantitative - shj curves difference magnitude)
sse_diff = torch.zeros(len(pts))
shj_diff = torch.tensor([
    torch.sum((shj[:, 1] - shj[:, 0])),
    torch.sum(shj[:, 2:5].mean(axis=1) - (shj[:, 1])),
    torch.sum((shj[:, 5] - shj[:, 2:5].mean(axis=1)))])

# include types 3-5 differences? should be low.. lower but not nth..
# shj_diff = torch.tensor([
#     torch.sum((shj[:, 1] - shj[:, 0])),
#     torch.sum(shj[:, 2:5].mean(axis=1) - (shj[:, 1])),
#     torch.sum((shj[:, 5] - shj[:, 2:5].mean(axis=1))),
#     torch.sum(torch.abs(shj[:, 2] - shj[:, 3])),
#     torch.sum(torch.abs(shj[:, 2] - shj[:, 4])),
#     torch.sum(torch.abs(shj[:, 3] - shj[:, 4]))])
# or assume diffs between them 0
# shj_diff = torch.tensor([
#     torch.sum((shj[:, 1] - shj[:, 0])),
#     torch.sum(shj[:, 2:5].mean(axis=1) - (shj[:, 1])),
#     torch.sum((shj[:, 5] - shj[:, 2:5].mean(axis=1))),
#     0, 0, 0])

# squared diffs
# - above seems better - maybe the direction does matter..
# shj_diff = torch.tensor([
#     torch.sum(torch.square((shj[:, 1] - shj[:, 0]))),
#     torch.sum(torch.square(shj[:, 2:5].mean(axis=1) - (shj[:, 1]))),
#     torch.sum(torch.square((shj[:, 5] - shj[:, 2:5].mean(axis=1))))])

# shj_diff = torch.tensor([
#     torch.sum(torch.square((shj[:, 1] - shj[:, 0]))),
#     torch.sum(torch.square(shj[:, 2:5].mean(axis=1) - (shj[:, 1]))),
#     torch.sum(torch.square((shj[:, 5] - shj[:, 2:5].mean(axis=1)))),
#     torch.sum(torch.square(torch.abs(shj[:, 2] - shj[:, 3]))),
#     torch.sum(torch.square(torch.abs(shj[:, 2] - shj[:, 4]))),
#     torch.sum(torch.square(torch.abs(shj[:, 3] - shj[:, 4])))])
# # or assume diffs between them 0
# shj_diff = torch.tensor([
#     torch.sum(torch.square((shj[:, 1] - shj[:, 0]))),
#     torch.sum(torch.square(shj[:, 2:5].mean(axis=1) - (shj[:, 1]))),
#     torch.sum(torch.square((shj[:, 5] - shj[:, 2:5].mean(axis=1)))),
#     0, 0, 0])

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

    # difference between curves magnitude
    diff = torch.tensor([
        torch.sum(pts[iparam][1] - pts[iparam][0]),
        torch.sum(pts[iparam][2:5].mean(axis=0) - pts[iparam][1]),
        torch.sum(pts[iparam][5] - pts[iparam][2:5].mean(axis=0))])
    # include diffs of 3-5
    # diff = torch.tensor([
    #     torch.sum(pts[iparam][1] - pts[iparam][0]),
    #     torch.sum(pts[iparam][2:5].mean(axis=0) - pts[iparam][1]),
    #     torch.sum(pts[iparam][5] - pts[iparam][2:5].mean(axis=0)),
    #     torch.sum(torch.abs(pts[iparam][2] - pts[iparam][3])),
    #     torch.sum(torch.abs(pts[iparam][2] - pts[iparam][4])),
    #     torch.sum(torch.abs(pts[iparam][3] - pts[iparam][4]))
    #     ])

    # should it be squared (abs) differences for all?
    # diff = torch.tensor([
    #     torch.sum(torch.square(pts[iparam][1] - pts[iparam][0])),
    #     torch.sum(
    #         torch.square(pts[iparam][2:5].mean(axis=0) - pts[iparam][1])),
    #     torch.sum(
    #         torch.square(pts[iparam][5] - pts[iparam][2:5].mean(axis=0)))
    #     ])

    # diff = torch.tensor([
    #     torch.sum(torch.square(pts[iparam][1] - pts[iparam][0])),
    #     torch.sum(
    #         torch.square(pts[iparam][2:5].mean(axis=0) - pts[iparam][1])),
    #     torch.sum(
    #         torch.square(pts[iparam][5] - pts[iparam][2:5].mean(axis=0))),
    #     torch.sum(torch.square(pts[iparam][2] - pts[iparam][3])),
    #     torch.sum(torch.square(pts[iparam][2] - pts[iparam][4])),
    #     torch.sum(torch.square(pts[iparam][3] - pts[iparam][4]))
    #     ])

    sse_diff[iparam] = torch.sum(torch.square(diff - shj_diff))

# criteria 1 already reduces to <12% of the params

ind_nll = nlls == nlls[ptn_criteria_1].min()
ind_sse = sse == sse[ptn_criteria_1].min()
ind_sse_diff = sse_diff == sse_diff[ptn_criteria_1].min()

# 2 sses weighted
# - with mse & mean all, w=.35-.4
# - with sse & sum all, w=.9. less then pr3 fast. more then too steep 6
w = .5 # larger = weight total more, smaller = weight differences more
sses_w = sse * w + sse_diff * (1-w)
# ind_sse_w = sses_w == sses_w[~sses_w.isnan()].min()  # ignore qual pattern
ind_sse_w = sses_w == sses_w[ptn_criteria_1].min()

# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
# print(param_sets[ind_nll])
print(param_sets[ind_sse])
print(param_sets[ind_sse_diff])

print(param_sets[ind_sse_w])

# plt.plot(nlls[ptn_criteria_1])
# # plt.ylim([88, 97])
# plt.show()
# plt.plot(sse[ptn_criteria_1])
# # plt.ylim([0, 9])
# plt.show()
# plt.plot(sses_w[ptn_criteria_1])
# plt.show()

# select which to use
ind = ind_sse_w

# more criteria
# - maybe faster type I / slower type VI
# - types III-V need to be more similar (e.g. within some range)
# - type I, II and the III-V's need to be larger differences

# %% plot

# matching differences, fitting differences of 3-5. w=.5
# - similar to .9 curve
# tensor([[0.4000, 3.0000, 0.4500, 0.2500, 0.3500, 0.9000]])
# matching differences, fitting differences of 3-5. w=.9
# - type 3 faster, 4 is slower - fitting curve more
# tensor([[0.4000, 7.0000, 0.5500, 0.0500, 0.4500, 0.9000]])
# matching differences, fitting differences of 3-5. w=.95
# tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]])

# matching differences, with equality of 3-5 (assuming 0). w=.5
# - type 3 is faster here
# tensor([[0.4000, 3.0000, 0.6500, 0.2500, 0.4500, 0.9000]])
# matching differences, with equality of 3-5 (assuming 0). w=.8
# - similar to above
# tensor([[0.4000, 7.0000, 0.9500, 0.0500, 0.3500, 0.9000]])
# matching differences, with equality of 3-5 (assuming 0). w=.9
# tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]])

# matching differences, not equality of 3-5. w=.25
# - less than .5 is too slow and 2 and 3 stuck together
# tensor([[0.4000, 3.0000, 0.1500, 0.1500, 0.1500, 0.9000]])
# matching differences, not equality of 3-5. w=.5 - gd, with type 3 faster
# tensor([[0.4000, 3.0000, 0.6500, 0.2500, 0.4500, 0.9000]])
# matching differences, not equality of 3-5. w=.95
# - needs to be .95 for 3-5 to stick together
# tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]])


# new - squared differences for matching the differences
# - all the same...? not caring about the differences much
# matching differences, fitting differences of 3-5. w=.5
# tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]])
# matching differences, with equality of 3-5 (assuming 0). w=.5
# - all 3 rulex's together
# - actully, the weighting doesn't change anything
# tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]])


saveplots = True

fntsiz = 15
ylims = (0., .55)

import matplotlib.font_manager as font_manager
# for roman numerals
font = font_manager.FontProperties(family='Tahoma',
                                   style='normal', size=fntsiz-2)

fig, ax = plt.subplots(2, 1)
ax[0].plot(shj)
ax[0].set_ylim(ylims)
ax[0].set_aspect(17)
ax[0].legend(('I', 'II', 'III', 'IV', 'V', 'VI'), fontsize=7)
ax[1].plot(pts[ind].T.squeeze())
ax[1].set_ylim(ylims)
ax[1].set_aspect(17)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'shj_gsearch_n94_subplots_{}units_k{}_w{}_notfitrulexdiffs.pdf'
                           .format(
                               n_units, k, w))
    plt.savefig(figname)
plt.show()

# best params by itself
fig, ax = plt.subplots(1, 1)
ax.plot(pts[ind].T.squeeze())
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
ax.legend(('I', 'II', 'III', 'IV', 'V', 'VI'), prop=font)
plt.tight_layout()
if saveplots:
    figname = os.path.join(figdir,
                           'shj_gsearch_{}units_k{}_w{}_notfitrulexdiffs.pdf'.format(
                               n_units, k, w))
    plt.savefig(figname)
plt.show()

# nosofsky '94 by itself
fig, ax = plt.subplots(1, 1)
ax.plot(shj)
ax.set_ylim(ylims)
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
# ax.legend(('1', '2', '3', '4', '5', '6'), fontsize=fntsiz-2)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
plt.tight_layout()
# if saveplots:
#     figname = os.path.join(figdir, 'nosofsky94_shj.pdf')
#     plt.savefig(figname)
plt.show()

# plot on top of each other
fig, ax = plt.subplots(1, 1)
ax.plot(shj, 'k')
ax.plot(pts[ind].T.squeeze(), 'o-')
ax.tick_params(axis='x', labelsize=fntsiz-3)
ax.tick_params(axis='y', labelsize=fntsiz-3)
ax.set_ylim(ylims)
ax.set_xlabel('Learning Block', fontsize=fntsiz)
ax.set_ylabel('Probability of Error', fontsize=fntsiz)
plt.show()


 # %% run MLE after gridsearch
import sys
import numpy as np
import time
from scipy import stats
from scipy import optimize as opt

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
sys.path.append('/Users/robert.mok/Documents/GitHub/multiunit-cluster')
from MultiUnitCluster import (MultiUnitCluster, train)

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
                 [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]]

behavior_sequence = shj.T

sim_info = {
    'n_units': n_units,
    'attn_type': 'dimensional_local',
    'k': k,
    'niter': 20  # niter
    }

# set seeds for niters of shj problem randomised - same seqs across params
seeds = torch.arange(sim_info['niter'])*10

# params
# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
# - can include multiple starts from various above
# starts = [param_sets[ind]]
starts = [[1.8000, 1.0000, 0.5500, 0.1500, 0.1500, 0.9000]]

bounds = [(0., 3.), (0, 20), (0, 1), (0, 1), (0, 1), (0, 1)]


def negloglik(model_pr, behavior_sequence):
    return -np.sum(stats.norm.logpdf(behavior_sequence, loc=model_pr))


# TODO add weighted for two things to fit

def sumsqerr(model_pr, behavior_sequence):
    torch.sum(torch.square(model_pr - behavior_sequence))



def run_model_shj(
        start_params, sim_info=sim_info, beh_seq=behavior_sequence):

    nll_all = torch.zeros(6)

    # run multiple iterations
    pt_all = torch.zeros([sim_info['niter'], 6, 16])
    for i in range(sim_info['niter']):

        # six problems
        for problem in range(6):

            stim = six_problems[problem]
            stim = torch.tensor(stim, dtype=torch.float)
            inputs = stim[:, 0:-1]
            output = stim[:, -1].long()  # integer
            # 16 per block
            inputs = inputs.repeat(2, 1)
            output = output.repeat(2).T

            # initialize model
            model = MultiUnitCluster(sim_info['n_units'], 3,
                                     sim_info['attn_type'],
                                     sim_info['k'],
                                     params=None,
                                     fit_params=True,
                                     start_params=start_params)

            model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
                model, inputs, output, 16, shuffle=True, shuffle_seed=seeds[i],
                shj_order=True)

            pt_all[i, problem] = 1 - epoch_ptarget.detach()

    for problem in range(6):
        nll_all[problem] = negloglik(pt_all[:, problem].mean(axis=0),
                                     beh_seq[problem])

        print(nll_all[problem])
    print(nll_all.sum())
    return nll_all.sum()


def fit_mle(func, starts, bounds=None):

    # looping through starting parameters
    nll = float('inf')
    res = []
    method = ['SLSQP', 'L-BFGS-B', 'Nelder-Mead', 'BFGS'][1]

    counter = 0
    for start_params in starts:
        res = opt.minimize(func, start_params, method=method, bounds=bounds) #, options={'maxfev': 150}) #, bounds=bounds)
        if res.fun < nll:  # if new result is smaller, replace it
            nll = res.fun
            bestparams = res
            print(nll)
            print(bestparams)
        print(counter)
        counter += 1
    return bestparams


t0 = time.time()
bestparams = fit_mle(run_model_shj, starts, bounds)
t1 = time.time()
print(t1-t0)
