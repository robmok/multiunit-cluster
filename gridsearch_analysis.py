#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:57:19 2021

Analysis on grid search results

- First a rough grid search (fingsearch=False) then a finer gridsearch
(finegsearch=True)
- standard SHJ (nbanks=False); or simulation with two banks of units modelling
anterior and posterior hpc (nbanks=True)

@author: robert.mok
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import itertools as it
import time

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/muc-results-all'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

finegsearch = False
nbanks = False

k = 0.005
n_units = 10000

# gsearch split into how many sets to load in
if not nbanks and not finegsearch:
    n_sets = 400  # gsearch
    resdir = os.path.join(maindir,
                      'muc-shj-gridsearch/gsearch_k{}_{}units_dist_final'.format(
    k, n_units))

elif not nbanks and finegsearch:  # set57 not complete? get
    n_sets = 350  # finegsearch
    resdir = os.path.join(maindir,
                          'muc-shj-gridsearch/finegsearch_k{}_{}units_dist_final'.format(
        k, n_units))

elif nbanks and not finegsearch:
    n_sets = 400
    resdir = os.path.join(maindir, 'muc-shj-gridsearch/gsearch_nbanks_final')

elif nbanks and finegsearch:
    n_sets = 400
    resdir = os.path.join(
        maindir, 'muc-shj-gridsearch/finegsearch_nbanks_final')

# SHJ
if not nbanks and not finegsearch:
    ranges = ([torch.arange(.2, 2.1, .2),
              torch.arange(1., 15., 2),
              torch.arange(1., 3.76, .25),  # more attn
              torch.arange(.05, .76, .1),  # / lr_scale,
              torch.arange(.05, 1., .1),
              torch.arange(.1, 1., .2)]
              )

# finegsearch
elif not nbanks and finegsearch:
    ranges = ([torch.arange(.1, .71, .1),
          torch.hstack([torch.arange(3., 9., 1), torch.arange(11., 15., 1)]),
          torch.arange(.75, 3.01, .25),
          torch.hstack([torch.arange(.025, .125, .025),
                        torch.arange(.3, .43, .025)]), # / lr_scale,
          torch.arange(.05, .7, .1),  # 1 less
          torch.arange(.5, 1., .2)]  # 1 extra
          )

# nbanks
if nbanks and not finegsearch:
    ranges = ([torch.arange(.1, .7, .1),
          torch.arange(.75, 2.5, .375),  # phi diff as 2 banks, no need so big
          torch.arange(.01, 3., .4),
          torch.arange(.01, 1., .15),
          torch.tensor([.3]),
          torch.tensor([.7]),  # .8 before

          torch.arange(1.8, 2.5, .1),
          torch.arange(.75, 2.5, .375),
          torch.arange(.001, .1, .05),  # 2 vals only
          torch.arange(.01, .4, .15),
          torch.tensor([.3]),
          torch.tensor([.7])]
          )
elif nbanks and finegsearch:
    ranges = ([torch.arange(.4, .9, .1),  # added .4
              # torch.arange(1., 1.251, .125),  # or just stick to 1.125
              torch.arange(.75, 1.251, .125), #  2 more than above
              torch.arange(.8, 2.2, .25),  # 10 rather than 6
              torch.arange(.55, .81, .05),
              torch.tensor([.3]),
              torch.tensor([.5, .7]),
    
              torch.arange(1.7, 2.2, .1), # 6 rather than 8
              torch.arange(2, 3.1, .25),  # 5 as before
              torch.tensor([.001]), # 1 instead of 2
              torch.tensor([.01]),  # 1 instead of 3
              torch.tensor([.3]),
              torch.tensor([.5, .7])]
              )

param_sets = torch.tensor(list(it.product(*ranges)))

sets = torch.arange(n_sets)

# load in
pts = []
nlls = []
recs = []
seeds = []

for iset in sets:

    if not nbanks and not finegsearch:
        fn = os.path.join(
            resdir,
            'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    elif not nbanks and finegsearch:
        fn = os.path.join(
            resdir,
            'shj_finegsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    elif nbanks and not finegsearch:
        fn = os.path.join(
            resdir,
            'shj_nbanks_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units,
                                                              iset))

    elif nbanks and finegsearch:
        fn = os.path.join(
            resdir,
            'shj_nbanks_finegsearch_k{}_{}units_set{}.pkl'.format(k, n_units,
                                                                  iset))

    # load - list: [nlls, pt_all, rec_all, seeds_all]
    open_file = open(fn, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    # print(loaded_list[0])  # check whats in them, any missing

    # if not loaded_list[1]:
    #     print(iset)

    if not loaded_list[0][-1]:  # check last one
    # if not torch.tensor(loaded_list[0][-1], dtype=torch.bool):  # check last one
        print(iset)

    nlls.extend(loaded_list[0])
    pts.extend(loaded_list[1])
    # recs.extend(loaded_list[2])
    # seeds.extend(loaded_list[3])


pts = torch.stack(pts)
nlls = torch.stack(nlls)
# recs = torch.stack(recs)
# seeds = torch.stack(seeds)

# nbanks - just get full model output
if nbanks:
    pts_banks = pts[:, :, 1:]  # get banks
    pts = pts[:, :, 0]  # full model - so can keep script like orig
# %% fit

t0 = time.time()

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

# criteria for fit:x

# quantitative:
# - compute SSE for each fit (already have nll)
# - compute SSE for differences

# qualitiative
# - pattern - 1, 2, 3-4-5, 6 (where 3, 4, 5 can be in any order for now). all
# points have to be faster (for now - maybe do 80% points if problem?)

# iparam = 0

# match threshold
# - 1 if match fully. can allow some error to be safe, eg ~.9/.95
# - note depending on the comparison, num total is diff (so prop is diff)
match_thresh = .9

# criterion 1 - shj pattern (qualitative)
sse = torch.zeros(len(pts))
ptn_criteria_1 = torch.zeros(len(pts), dtype=torch.bool)

# criterion 2 (quantitative - shj curves difference magnitude)
sse_diff = torch.zeros(len(pts))

# nbanks pattern
ptn_criteria_1_nbanks = torch.zeros(len(pts), 2, dtype=torch.bool)
ptn_criteria_2_nbanks = torch.zeros(len(pts), dtype=torch.bool)
ptn_criteria_3_nbanks = torch.zeros(len(pts), dtype=torch.bool)


# assume diffs between problems 3-5 0
shj_diff = torch.tensor([
    torch.sum(torch.square((shj[:, 1] - shj[:, 0]))),
    torch.sum(torch.square(shj[:, 2:5].mean(axis=1) - (shj[:, 1]))),
    torch.sum(torch.square((shj[:, 5] - shj[:, 2:5].mean(axis=1)))),
    0, 0, 0])

# separate all, average 3-5
sse_diff = torch.zeros([len(pts), 4])

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

    # nbanks -set some critiera
    # - all same, apart for 345 - thresh=if diff <.005 per blk, same
    if nbanks:
        thr_bs = .01  # .005
        for ibank in range(2):
            ptn = (torch.abs(pts_banks[iparam, :, ibank][0]
                              - pts_banks[iparam, :, ibank][1]) < thr_bs)
            ptn_bs_c1 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
            # 2 vs 3,4,5 same
            ptn = (torch.abs(pts_banks[iparam, :, ibank][1]
                              - pts_banks[iparam, :, ibank][2]) < thr_bs)
            ptn_bs_c2 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
            ptn = (torch.abs(pts_banks[iparam, :, ibank][1]
                              - pts_banks[iparam, :, ibank][3]) < thr_bs)
            ptn_bs_c3 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
            ptn = (torch.abs(pts_banks[iparam, :, ibank][1]
                              - pts_banks[iparam, :, ibank][4]) < thr_bs)
            ptn_bs_c4 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
            # 3, 4, 5 vs 6 same
            ptn = (torch.abs(pts_banks[iparam, :, ibank][2]
                              - pts_banks[iparam, :, ibank][5]) < thr_bs)
            ptn_bs_c5 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
            ptn = (torch.abs(pts_banks[iparam, :, ibank][3]
                              - pts_banks[iparam, :, ibank][5]) < thr_bs)
            ptn_bs_c6 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
            ptn = (torch.abs(pts_banks[iparam, :, ibank][4]
                              - pts_banks[iparam, :, ibank][5]) < thr_bs)
            ptn_bs_c7 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    
            ptn_criteria_1_nbanks[iparam, ibank] = (
                ptn_bs_c1 & ptn_bs_c2 & ptn_bs_c3 & ptn_bs_c4 & ptn_bs_c5 &
                ptn_bs_c6 & ptn_bs_c7
                )
    
        # params where bank 2 has it flipped for 1 and 6
        ptn = pts_banks[iparam, :, 1][5] < pts_banks[iparam, :, 1][:5]  # VI fastst
        ptn_bs_c8 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
        # ptn = pts_banks[iparam, :, 1][1] > pts_banks[iparam, :, 1][2:6]  # type II 2nd slowest - maybe not nec
        # ptn_bs_c9 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
        ptn = pts_banks[iparam, :, 1][0] > pts_banks[iparam, :, 1][1:6]  # type I slowest
        ptn_bs_c10 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    
        ptn_criteria_2_nbanks[iparam] = ptn_bs_c8 & ptn_bs_c10
        # ptn_criteria_2_nbanks[iparam] = ptn_bs_c8 & ptn_bs_c9 & ptn_bs_c10
    
        # cross bank differences
        # type I, bank 1 faster than bank 2
        ptn = pts_banks[iparam, :, 0][0] < pts_banks[iparam, :, 1][0]
        ptn_bs_c11 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
        # type VI, bank 2 faster than bank 1
        ptn = pts_banks[iparam, :, 0][5] > pts_banks[iparam, :, 1][5]
        ptn_bs_c12 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    
        # type 345 bank 2 faster
        ptn = pts_banks[iparam, :, 0][2:5] > pts_banks[iparam, :, 1][2:5]
        ptn_bs_c13 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    
        ptn_criteria_3_nbanks[iparam] = ptn_bs_c11 & ptn_bs_c12 & ptn_bs_c13

    diff = torch.tensor([
        torch.sum(torch.square(pts[iparam][1] - pts[iparam][0])),
        torch.sum(
            torch.square(pts[iparam][2:5].mean(axis=0) - pts[iparam][1])),
        torch.sum(
            torch.square(pts[iparam][5] - pts[iparam][2:5].mean(axis=0))),
        torch.sum(torch.square(pts[iparam][2] - pts[iparam][3])),
        torch.sum(torch.square(pts[iparam][2] - pts[iparam][4])),
        torch.sum(torch.square(pts[iparam][3] - pts[iparam][4]))
        ])

    sse_diff[iparam] = torch.sum(torch.square(diff - shj_diff))


    # separate all, with 3-5 equal
    tmp = torch.square(diff - shj_diff)
    sse_diff[iparam] = torch.cat([tmp[0:3].view(3, 1),
                                  tmp[-3:].sum().view(1, 1)]).squeeze()

t1 = time.time()
print(t1-t0)

# criteria 1 already reduces to <12% of the params

# remove those with nans at all (i.e. still got a curve for those with nanmean)
ind_nan = np.isnan(nlls)
nlls[ind_nan] = np.inf
sse[ind_nan] = np.inf
sse_diff[ind_nan] = np.inf

# pattern criteria
ptn_criteria = ptn_criteria_1  # standard

# nbanks
if nbanks:
    ptn_criteria = (
        ptn_criteria_1
        & ~torch.all(ptn_criteria_1_nbanks, axis=1)  # nbanks - rmv if all same
        & ptn_criteria_2_nbanks  # 2nd bank flipped
        & ptn_criteria_3_nbanks   # across banks - type 1<1 and 6>6 for banks 1vs2
        )

ind_nll = nlls == nlls[ptn_criteria].min()
ind_sse = sse == sse[ptn_criteria].min()
ind_sse_diff = sse_diff == sse_diff[ptn_criteria].min()

# %% 2 sses weighted

if not nbanks and not finegsearch:

    # gsearch
    # tensor([[0.4000, 7.0000, 1.0000, 0.0500, 0.2500, 0.9000]])
    w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])  # already gd
    # tensor([[0.2000, 5.0000, 3.7500, 0.3500, 0.2500, 0.9000]])
    w = torch.tensor([1/5, 1/5, 1/5, 10/5, 100/5])

    # best params
    # # tensor([[ 0.2000, 13.0000,  2.0000,  0.0500,  0.3500,  0.7000]])
    # w = torch.tensor([1/5, 1/5, 100/5, 100/5, 1000/5])  # like orig but better
    # tensor([[ 0.2000, 13.0000,  2.0000,  0.0500,  0.3500,  0.9000]])
    # w = torch.tensor([1/5, 1/5, 100/5, 50/5, 1000/5])  # 3 ever so slightly closer to rulexes
    # tensor([[ 0.2000, 13.0000,  2.7500,  0.0500,  0.3500,  0.9000]])
    w = torch.tensor([1/5, 1/5, 100/5, 25/5, 1000/5]) # best - just touching rulexes
    # tensor([[0.4000, 7.0000, 1.2500, 0.0500, 0.4500, 0.9000]])
    # w = torch.tensor([1/5, 1/5, 100/5, 10/5, 1000/5])  # all 3 rulexes together but overall faster (6 is closer to rulexes)
    # w = torch.tensor([1/5, 1/5, 100/5, 1/5, 50/5])
    # # tensor([[0.2000, 5.0000, 3.7500, 0.3500, 0.2500, 0.9000]])
    # w = torch.tensor([1/5, 1/5, 10/5, 10/5, 100/5])
    
    # --> to search in finegsearch
    # best: # tensor([[ 0.2000, 13.0000,  2.7500,  0.0500,  0.3500,  0.9000]])
    # tensor([[ 0.2/.4, 5/7/13,  2/1.25/2.75,  0.05/0.35,  .25/.35/.45,  [0.5?]/0.7/0.9]])

elif not nbanks and finegsearch:

    # finegsearch
    # tensor([[ 0.2000, 14.0000,  2.7500,  0.0500,  0.3500,  0.9000]])
    w = torch.tensor([1/5, 1/5, 3/5, 1/5, 1/5])  # best - all v sim to this from 3/5 for 3rd one


elif nbanks and not finegsearch:

    # tensor([[6.0000e-01, 1.1250e+00, 1.6100e+00, 6.1000e-01, 3.0000e-01, 7.0000e-01,
             # 2.0000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 7.0000e-01]])
    # w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])  # looks pretty gd already
    
    # tensor([[5.0000e-01, 1.1250e+00, 8.1000e-01, 7.6000e-01, 3.0000e-01, 7.0000e-01,
             # 1.9000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 7.0000e-01]])
    w = torch.tensor([1/5, 1/5, 1/5, 20/5, 1/5])  # better separation - best -  if param 4 goes up, stays the same

elif nbanks and finegsearch:

    # # tensor([[6.0000e-01, 1.1250e+00, 1.5500e+00, 7.0000e-01, 3.0000e-01, 8.0000e-01,
    #          # 1.7000e+00, 2.5000e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
    # w = torch.tensor([1/5, 1/5, 1/5, 150/5, 1/5])
    # # tensor([[6.0000e-01, 1.2500e+00, 8.0000e-01, 6.5000e-01, 3.0000e-01, 7.0000e-01,
    # #          1.7000e+00, 2.7500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
    # w = torch.tensor([1/5, 1/5, 1/5, 250/5, 1/5]) # gd
    # # tensor([[5.0000e-01, 1.2500e+00, 8.0000e-01, 7.0000e-01, 3.0000e-01, 7.0000e-01,
    # #          1.8000e+00, 2.5000e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
    # w = torch.tensor([1/5, 1/5, 1/5, 30/5, 1/5])
    # # tensor([[6.0000e-01, 1.0000e+00, 1.5500e+00, 7.0000e-01, 3.0000e-01, 8.0000e-01,
    # #          1.7000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
    w = torch.tensor([1/5, 1/5, 350/5, 350/5, 1/5]) # good - this is what's in the figure now

w = w / w.sum()
sses_w = sse * w[0] + torch.sum(sse_diff * w[1:], axis=1)
ind_sse_w = sses_w == sses_w[ptn_criteria].min()

if len(torch.nonzero(ind_sse_w)) > 1:
    ind_sse_w[torch.nonzero(ind_sse_w)[1]] = 0

print(param_sets[ind_sse_w])

# select which to use
ind = ind_sse_w

# if more than 1 match, remove one
# ind[torch.nonzero(ind_nll)[0]] = True
# ind[torch.nonzero(ind_nll)[1]] = False

# plot
w_str = ""
for iw in w[:-1]:
    w_str += str(np.around(iw.item(), decimals=3)) + '-'
# last one no '-'
w_str += str(np.around(w[-1].item(), decimals=3))

saveplots = False

fntsiz = 15
ylims = (0., .55)

import matplotlib.font_manager as font_manager
# for roman numerals
font = font_manager.FontProperties(family='Tahoma', style='normal',
                                   size=fntsiz-2)

# single bank plot
if not nbanks:
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
                                'shj_gsearch_n94_subplots_{}units_k{}_w{}.pdf'
                                .format(
                                    n_units, k, w_str))
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
                               'shj_gsearch_{}units_k{}_w{}.pdf'.format(
                                   n_units, k, w_str))
        plt.savefig(figname)
    plt.show()

# nbanks plt
else:

    fig, ax = plt.subplots(1, 1)
    ax.plot(pts[ind].T.squeeze())
    ax.tick_params(axis='x', labelsize=fntsiz-3)
    ax.tick_params(axis='y', labelsize=fntsiz-3)
    ax.set_ylim(ylims)
    ax.set_title('Full Model Output', fontsize=fntsiz)
    ax.set_xlabel('Learning Block', fontsize=fntsiz)
    ax.set_ylabel('Probability of Error', fontsize=fntsiz)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(pts_banks[ind, :, 0].T.squeeze())
    ax.tick_params(axis='x', labelsize=fntsiz-3)
    ax.tick_params(axis='y', labelsize=fntsiz-3)
    ax.set_ylim(ylims)
    ax.set_title('Module 1', fontsize=fntsiz)
    ax.set_xlabel('Learning Block', fontsize=fntsiz)
    ax.set_ylabel('Probability of Error', fontsize=fntsiz)
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(pts_banks[ind, :, 1].T.squeeze())
    ax.tick_params(axis='x', labelsize=fntsiz-3)
    ax.tick_params(axis='y', labelsize=fntsiz-3)
    ax.set_ylim(ylims)
    ax.set_title('Module 2', fontsize=fntsiz)
    ax.set_xlabel('Learning Block', fontsize=fntsiz)
    ax.set_ylabel('Probability of Error', fontsize=fntsiz)
    plt.tight_layout()
    plt.show()
    
    aspect = 1.5
    linewidth = 1.
    gridalpha = .75
    bgcol1 = np.array([31, 119, 180])/255  # blue
    bgcol2 = np.array([255, 240, 0])/255  # yellow
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(pts[ind].T.squeeze(), linewidth=linewidth)
    ax[0].set_ylim(ylims)
    ax[0].set_box_aspect(aspect)
    ax[0].set_title('Output', fontsize=fntsiz)
    ax[0].set_ylabel('Probability of Error', fontsize=fntsiz)
    ax[0].tick_params(axis='x', labelsize=fntsiz-2)
    ax[0].tick_params(axis='y', labelsize=fntsiz-2)
    ax[0].grid(linestyle='--', alpha=gridalpha)
    ax[1].plot(pts_banks[ind, :, 0].T.squeeze(), linewidth=linewidth)
    ax[1].set_ylim(ylims)
    ax[1].set_xlabel('Block', fontsize=fntsiz)
    ax[1].set_box_aspect(aspect)
    ax[1].set_title('Anterior HPC', fontsize=fntsiz)
    labels = ['', '', '', '', '', '']
    ax[1].set_yticklabels(labels)
    ax[1].tick_params(axis='x', labelsize=fntsiz-2)
    ax[1].grid(linestyle='--', alpha=gridalpha)
    ax[1].set_facecolor(np.append(bgcol1, 0.15))  # add alpha
    ax[2].plot(pts_banks[ind, :, 1].T.squeeze(), linewidth=linewidth)
    ax[2].set_ylim(ylims)
    ax[2].set_box_aspect(aspect)
    labels = ['', '', '', '', '', '']
    ax[2].set_yticklabels(labels)
    ax[2].tick_params(axis='x', labelsize=fntsiz-2)
    ax[2].set_title('Posterior HPC', fontsize=fntsiz)
    ax[2].grid(linestyle='--', alpha=gridalpha)
    ax[2].legend(('I', 'II', 'III', 'IV', 'V', 'VI'), fontsize=10)
    ax[2].set_facecolor(np.append(bgcol2, 0.15))  # add alpha
    plt.tight_layout()
    if saveplots:
        figname = os.path.join(figdir,'shj_nbanks_curves_sep_b12_gsearch_cols.pdf')
        plt.savefig(figname)
    plt.show()

# compare 2 banks
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(pts_banks[ind, :, 0].T.squeeze())
# ax[0].set_ylim(ylims)
# ax[1].plot(pts_banks[ind, :, 1].T.squeeze())
# ax[1].set_ylim(ylims)
# ax[1].legend(('I', 'II', 'III', 'IV', 'V', 'VI'), fontsize=10)
# plt.tight_layout()
# plt.show()

# # compare type 6 across banks - since small difference
# plt.plot(torch.stack([pts_banks[ind, 5, 0], pts_banks[ind, 5, 1]]).squeeze().T)
# plt.show()

# compare rulex across banks
# plt.plot((pts_banks[ind, 2:5, 0]-pts_banks[ind, 2:5, 1]).squeeze().T)
# plt.show()

# plt.plot(torch.stack([pts_banks[ind, 2, 0], pts_banks[ind, 2, 1]]).squeeze().T)
# plt.title('type III')
# plt.show()
# plt.plot(torch.stack([pts_banks[ind, 3, 0], pts_banks[ind, 3, 1]]).squeeze().T)
# plt.title('type IV')
# plt.show()
# plt.plot(torch.stack([pts_banks[ind, 4, 0], pts_banks[ind, 4, 1]]).squeeze().T)
# plt.title('type V')
# plt.show()


# # nosofsky '94 by itself
# fig, ax = plt.subplots(1, 1)
# ax.plot(shj)
# ax.set_ylim(ylims)
# ax.tick_params(axis='x', labelsize=fntsiz-3)
# ax.tick_params(axis='y', labelsize=fntsiz-3)
# # ax.legend(('1', '2', '3', '4', '5', '6'), fontsize=fntsiz-2)
# ax.set_xlabel('Learning Block', fontsize=fntsiz)
# ax.set_ylabel('Probability of Error', fontsize=fntsiz)
# plt.tight_layout()
# # if saveplots:
# #     figname = os.path.join(figdir, 'nosofsky94_shj.pdf')
# #     plt.savefig(figname)
# plt.show()

# # plot on top of each other
# fig, ax = plt.subplots(1, 1)
# ax.plot(shj, 'k')
# ax.plot(pts[ind].T.squeeze(), 'o-')
# ax.tick_params(axis='x', labelsize=fntsiz-3)
# ax.tick_params(axis='y', labelsize=fntsiz-3)
# ax.set_ylim(ylims)
# ax.set_xlabel('Learning Block', fontsize=fntsiz)
# ax.set_ylabel('Probability of Error', fontsize=fntsiz)
# plt.show()

