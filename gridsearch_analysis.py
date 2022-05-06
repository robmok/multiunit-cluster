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
import time

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'
figdir = os.path.join(maindir, 'multiunit-cluster_figs')

k = 0.01
n_units = 2000

# gsearch split into how many sets to load in
# 450 sets. 440 for finegsearch distsq1. 348 for finegsearch dist. 349 distsq
# new 2022: 400 sets
# finegsearch distsq2 349 sets. finegsearch dist1 400 sets
# nbanks 360, fine 128
n_sets = 400

# resdir = os.path.join(maindir,
#                       'muc-shj-gridsearch/gsearch_k{}_{}units'.format(
#     k, n_units))

resdir = os.path.join(maindir,
                      'muc-shj-gridsearch/gsearch_k{}_{}units_dist'.format(
    k, n_units))

# resdir = os.path.join(maindir,
#                       'muc-shj-gridsearch/gsearch_k{}_{}units_distsq'.format(
#     k, n_units))

resdir = os.path.join(
    maindir, 'muc-shj-gridsearch/finegsearch_k{}_{}units_dist1'.format(
        k, n_units))

# resdir = os.path.join(
#     maindir, 'muc-shj-gridsearch/finegsearch_k{}_{}units_distsq2'.format(
#         k, n_units))


# large attn - 350 sets
# resdir = os.path.join(maindir,
#                       'muc-shj-gridsearch/gsearch_k{}_{}units_dist_large_attn'.format(
#     k, n_units))

# resdir = os.path.join(
#     maindir, 'muc-shj-gridsearch/finegsearch_k{}_{}units_dist3_attn'.format(
#         k, n_units))

# large attn redo, shj order
resdir = os.path.join(maindir,
                      'muc-shj-gridsearch/gsearch_k{}_{}units_dist_final'.format(
    k, n_units))


# # nbanks
# resdir = os.path.join(
#     maindir, 'muc-shj-gridsearch/gsearch_nbanks')


# resdir = os.path.join(
#     maindir, 'muc-shj-gridsearch/finegsearch_nbanks')


ranges = ([torch.arange(.4, 2.1, .2),
          torch.arange(1., 15., 2),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

# newer ~August/Sept dist
ranges = ([torch.arange(.2, 2.1, .2),
          torch.arange(1., 15., 2),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

# when changing dist**2, changing c to start from .3, which loses one c value
ranges = ([torch.arange(.3, 2.1, .2),
          torch.arange(1., 15., 2),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

# # finegridsearch distsq 1
# ranges = ([torch.arange(.2, .45, 1/30),
#           torch.arange(1., 5.1, .25),
#           torch.arange(.15, .66, .1),
#           torch.arange(.15, .8, .05),
#           torch.arange(.25, .5, .1),
#           torch.arange(.8, 1.01, .1)]
#           )

# dist
# c=.2/1.2, phi=3/1, lr_attn=.75/.95/, lr_nn=.35/.85, lr_clus=.35/.45, group=.9
ranges = ([torch.cat([torch.arange(.1, .35, .05),
                      torch.arange(1.05, 1.25, .05)]),
          torch.arange(.75, 5.1, .25),
          torch.arange(.55, .96, .1),
          torch.cat([torch.arange(.25, .5, .05),
                      torch.arange(.8, .96, .05)]),
          torch.tensor([.25, .35, .45]),
          torch.tensor([.9])]
          )

# # distsq1
# # c=0.3/~.1, phi=1/3/5, lr_attn=.05/.15/.35, lr_nn=.05/.15/.25, lr_clus=.15/.45. lr_group .3/.9
# ranges = ([torch.arange(.1, .45, .05),
#           torch.arange(.75, 5.6, .25),  # many
#           torch.arange(.05, .41, .05),
#           torch.arange(.05, .36, .05),
#           torch.tensor([.15, .35, .45]),
#           torch.tensor([.3, .9])]
#           )

# distsq2
ranges = ([torch.arange(.4, 1.2, .2),
          torch.arange(.25, 2., .25),
          torch.arange(.15, .66, .1),
          torch.arange(.25, .76, .05),
          torch.arange(.35, .8, .1),
          torch.arange(.1, .5, .1)]
          )

# dist1
ranges = ([torch.arange(.2, .8, .1),
            torch.arange(.75, 3., .25),
            torch.arange(.25, .96, .1),
            torch.arange(.15, .65, .1),
            torch.arange(.25, .56, .1),
            torch.arange(.6, 1., .1)]
          )

# dist - with attn lr > 1., with fewer params
ranges = ([torch.arange(.2, 1.1, .2),
          torch.arange(1., 11., 2),
          torch.arange(1., 2.76, .25),
          torch.arange(.05, .76, .1),
          torch.arange(.15, .76, .1),
          torch.arange(.5, 1., .2)]
          )

# finegsearch attn
ranges = ([torch.arange(.2, .8, .1),
      torch.arange(3., 10., 1),
      torch.arange(2., 4.01, .25),
      torch.arange(.005, .15, .02),
      torch.arange(.25, .66, .1),
      torch.arange(.7, 1., .2)]
      )

# redo with shj order - final
ranges = ([torch.arange(.2, 2.1, .2),
          torch.arange(1., 15., 2),
          torch.arange(1., 2.76, .25),
          torch.arange(.05, .76, .1),  # / lr_scale,
          torch.arange(.05, 1., .1),
          torch.arange(.1, 1., .2)]
          )

# # nbanks
# ranges = ([torch.arange(.1, .7, .1),
#           torch.arange(.75, 2.5, .375),
#           torch.arange(.01, 3., .4),
#           torch.arange(.01, 1., .15),
#           torch.tensor([.3]),
#           torch.tensor([.8]),

#           torch.arange(1.8, 2.5, .1),
#           torch.arange(.75, 2.5, .375),
#           torch.arange(.001, .1, .05),  # 2 vals only
#           torch.arange(.01, .4, .15),
#           torch.tensor([.3]),
#           torch.tensor([.8])]
#           )

# # finegsearch nbanks
# ranges = ([torch.arange(.5, .9, .1),
#           torch.arange(.75, 1.251, .125), #  2 more than above
#           torch.arange(.8, 2.2, .25),
#           torch.arange(.55, .81, .05),  #
#           torch.tensor([.3]),
#           torch.tensor([.5, .8]),  # group lr

#           torch.arange(1.7, 2.2, .1),
#           torch.arange(2, 3.1, .25),
#           torch.tensor([.001]),
#           torch.tensor([.01]),
#           torch.tensor([.3]),
#           torch.tensor([.5, .8])]
#           )

param_sets = torch.tensor(list(it.product(*ranges)))

sets = torch.arange(n_sets)

# TMP
# sets = sets[(sets != 80) & (sets != 109)]  # TMP remove sets 80 and 109
# sets = sets[(sets != 57)]
# sets = sets[(sets != 156)]
# sets = sets[(sets != 68) & (sets != 69) & (sets != 116)]

# # TMP - remove some sets if incomplete
# sets = torch.arange(0, len(param_sets)+1, 700)
# ind = torch.ones(len(param_sets), dtype=torch.bool)
# ind[sets[68]:sets[70]] = False
# ind[sets[116]:sets[117]] = False
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

    # fn = os.path.join(
    #     resdir,
    #     'shj_finegsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    # fn = os.path.join(
    #     resdir,
    #     'shj_nbanks_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    # fn = os.path.join(
    #     resdir,
    #     'shj_nbanks_finegsearch_k{}_{}units_set{}.pkl'.format(k, n_units,
    #                                                           iset))

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

# after doing nan mean, these are now numpy arrays for dist. will change later (changed for dist**2)
# pts = torch.tensor(np.stack(pts))
# nlls = torch.tensor(np.stack(nlls))

# # nbanks - just get full model output for now
# pts_banks = pts[:, :, 1:]  # get banks
# pts = pts[:, :, 0]  # full model - so can keep script like orig
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

# squared diffs
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
# or assume diffs between them 0
shj_diff = torch.tensor([
    torch.sum(torch.square((shj[:, 1] - shj[:, 0]))),
    torch.sum(torch.square(shj[:, 2:5].mean(axis=1) - (shj[:, 1]))),
    torch.sum(torch.square((shj[:, 5] - shj[:, 2:5].mean(axis=1)))),
    0, 0, 0])

# separate all, average 3-5
sse_diff = torch.zeros([len(pts), 4])

# # separate 2-3 diff, this is key one i can't get
# shj_diff = torch.tensor([
#     torch.sum(torch.square((shj[:, 1] - shj[:, 0]))),
#     torch.sum(torch.square(shj[:, 2] - (shj[:, 1]))),  # new
#     torch.sum(torch.square(shj[:, 2:5].mean(axis=1) - (shj[:, 1]))),
#     torch.sum(torch.square((shj[:, 5] - shj[:, 2:5].mean(axis=1)))),
#     0, 0, 0])

# sse_diff = torch.zeros([len(pts), 5])

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

    # # nbanks -set some critiera
    # # - all same, apart for 345 - thresh=if diff <.005 per blk, same
    # thr_bs = .01  # .005
    # for ibank in range(2):
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][0]
    #                      - pts_banks[iparam, :, ibank][1]) < thr_bs)
    #     ptn_bs_c1 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    #     # 2 vs 3,4,5 same
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][1]
    #                      - pts_banks[iparam, :, ibank][2]) < thr_bs)
    #     ptn_bs_c2 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][1]
    #                      - pts_banks[iparam, :, ibank][3]) < thr_bs)
    #     ptn_bs_c3 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][1]
    #                      - pts_banks[iparam, :, ibank][4]) < thr_bs)
    #     ptn_bs_c4 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    #     # 3, 4, 5 vs 6 same
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][2]
    #                      - pts_banks[iparam, :, ibank][5]) < thr_bs)
    #     ptn_bs_c5 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][3]
    #                      - pts_banks[iparam, :, ibank][5]) < thr_bs)
    #     ptn_bs_c6 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    #     ptn = (torch.abs(pts_banks[iparam, :, ibank][4]
    #                      - pts_banks[iparam, :, ibank][5]) < thr_bs)
    #     ptn_bs_c7 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh

    #     ptn_criteria_1_nbanks[iparam, ibank] = (
    #         ptn_bs_c1 & ptn_bs_c2 & ptn_bs_c3 & ptn_bs_c4 & ptn_bs_c5 &
    #         ptn_bs_c6 & ptn_bs_c7
    #         )

    # # params where bank 2 has it flipped for 1 and 6
    # ptn = pts_banks[iparam, :, 1][5] < pts_banks[iparam, :, 1][:5]  # VI fastst
    # ptn_bs_c8 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    # # ptn = pts_banks[iparam, :, 1][1] > pts_banks[iparam, :, 1][2:6]  # type II 2nd slowest - maybe not nec
    # # ptn_bs_c9 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    # ptn = pts_banks[iparam, :, 1][0] > pts_banks[iparam, :, 1][1:6]  # type I slowest
    # ptn_bs_c10 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh

    # ptn_criteria_2_nbanks[iparam] = ptn_bs_c8 & ptn_bs_c10
    # # ptn_criteria_2_nbanks[iparam] = ptn_bs_c8 & ptn_bs_c9 & ptn_bs_c10

    # # cross bank differences
    # # type I, bank 1 faster than bank 2
    # ptn = pts_banks[iparam, :, 0][0] < pts_banks[iparam, :, 1][0]
    # ptn_bs_c11 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    # # type VI, bank 2 faster than bank 1
    # ptn = pts_banks[iparam, :, 0][5] > pts_banks[iparam, :, 1][5]
    # ptn_bs_c12 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh

    # # type 345 bank 2 faster
    # ptn = pts_banks[iparam, :, 0][2:5] > pts_banks[iparam, :, 1][2:5]
    # ptn_bs_c13 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh

    # ptn_criteria_3_nbanks[iparam] = ptn_bs_c11 & ptn_bs_c12 & ptn_bs_c13






    # # squared (abs) differences
    # diff = torch.tensor([
    #     torch.sum(torch.square(pts[iparam][1] - pts[iparam][0])),
    #     torch.sum(
    #         torch.square(pts[iparam][2:5].mean(axis=0) - pts[iparam][1])),
    #     torch.sum(
    #         torch.square(pts[iparam][5] - pts[iparam][2:5].mean(axis=0)))
    #     ])

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

    # # add extra diff between type 2 and 3 - difficult to get
    # diff = torch.tensor([
    #     torch.sum(torch.square(pts[iparam][1] - pts[iparam][0])),
    #     torch.sum(torch.square(pts[iparam][2] - pts[iparam][1])),
    #     torch.sum(
    #         torch.square(pts[iparam][2:5].mean(axis=0) - pts[iparam][1])),
    #     torch.sum(
    #         torch.square(pts[iparam][5] - pts[iparam][2:5].mean(axis=0))),
    #     torch.sum(torch.square(pts[iparam][2] - pts[iparam][3])),
    #     torch.sum(torch.square(pts[iparam][2] - pts[iparam][4])),
    #     torch.sum(torch.square(pts[iparam][3] - pts[iparam][4]))
    #     ])

    # tmp = torch.square(diff - shj_diff)
    # sse_diff[iparam] = torch.cat([tmp[0:4].view(4, 1),
    #                               tmp[-3:].sum().view(1, 1)]).squeeze()

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
# ptn_criteria = (
#     ptn_criteria_1
#     & ~torch.all(ptn_criteria_1_nbanks, axis=1)  # nbanks - rmv if all same
#     & ptn_criteria_2_nbanks  # 2nd bank flipped
#     & ptn_criteria_3_nbanks   # across banks - type 1<1 and 6>6 for banks 1vs2
#     )

ind_nll = nlls == nlls[ptn_criteria].min()
ind_sse = sse == sse[ptn_criteria].min()
ind_sse_diff = sse_diff == sse_diff[ptn_criteria].min()

# ind_sse_diff1 = sse_diff1 == sse_diff1[ptn_criteria].min()
# ind_sse_diff2 = sse_diff2 == sse_diff2[ptn_criteria].min()


# %%

# # 2 sses weighted
# # - with mse & mean all, w=.35-.4
# # - with sse & sum all, w=.9. less then pr3 fast. more then too steep 6
# w = .5  # larger = weight total more, smaller = weight differences more
# sses_w = sse * w + sse_diff * (1-w)
# # ind_sse_w = sses_w == sses_w[~sses_w.isnan()].min()  # ignore qual pattern
# ind_sse_w = sses_w == sses_w[ptn_criteria].min()

# # 3 sses weighted - sse, sse_diff1 (1-2, 2-345, 345-6), sse_diff2 (3-5 equal)
# w = [1/3, 1/3, 1/3]
# w = [2/6, 1/6, 3/6]
# w = [1/6, 0/6, 5/6]

# w = [3/6, 1/6, 4/6]
# # w = np.array(w)/np.array(w).sum()

# sses_w = sse * w[0] + sse_diff1 * w[1] + sse_diff * w[2]
# ind_sse_w = sses_w == sses_w[ptn_criteria].min()


# separate all - 4 diffs. 1st is total sse, last is 3-5 equality

# dist
# tensor([[2.0000, 1.0000, 0.5500, 0.1500, 0.2500, 0.9000]])
w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])  # - actually ok - like prev plot, type 6 a bit fast
# tensor([[1.2000, 1.0000, 0.7500, 0.3500, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 2/5, 9/5, 1/5])  # v gd but bumpy, redo sims for this? i suspect recruits too many clus?

# tensor([[0.2000, 3.0000, 0.9500, 0.8500, 0.3500, 0.9000]])  # - ok but type 3 fast
# w = torch.tensor([1/5, 1/5, 300/5, 150/5, 1/5])
# w = torch.tensor([2/5, 1/5, 600/5, 300/5, 100/5]) # - same as above

# tensor([[1.8000, 1.0000, 0.5500, 0.1500, 0.3500, 0.7000]])  # type 6 too fast. c too high..
# w = torch.tensor([2/5, 1/5, 250/5, 100/5, 100/5])

# params to test:
# c=.2/1.2, phi=3/1, lr_attn=.75/.95/, lr_nn=.35/.85, lr_clus=.35/.45, group=.9
# - planning to run now


# dist**2
# tensor([[0.3000, 9.0000, 0.7500, 0.0500, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])  # type 3 too fast

# weight 345-6 more
# tensor([[1.1000, 1.0000, 0.3500, 0.3500, 0.1500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 5000/5])  # # not bad, 2-345 diff too small

# weight 2-345 an 345-6 simliarily, type 3 too slow
# tensor([[0.3000, 5.0000, 0.1500, 0.1500, 0.4500, 0.3000]])
# w = torch.tensor([2/5, 1/5, 200/5, 200/5, 100/5]) # 2-345 an 345-6 simliarily, type 3 too fast. but ok pattern

# weight 2-345 more
# tensor([[0.3000, 3.0000, 0.0500, 0.2500, 0.4500, 0.3000]])
# w = torch.tensor([1/5, 1/5, 1/5, 500/5, 1/5])  # gd pattern but all slow

# tensor([[0.9000, 3.0000, 0.1500, 0.0500, 0.7500, 0.3000]]) - lr_clus a bit fast?
# w = torch.tensor([1/5, 1/5, 200/5, 10/5, 100/5])  # this is best gd pattern, 6 still a bit fast

# params to test
# tensor([[0.3000, 9.0000, 0.7500, 0.0500, 0.4500, 0.9000]]) # without ws - ignore?
# tensor([[1.1000, 1.0000, 0.3500, 0.3500, 0.1500, 0.9000]])
# tensor([[0.3000, 5.0000, 0.1500, 0.1500, 0.4500, 0.3000]])
# tensor([[0.3000, 3.0000, 0.0500, 0.2500, 0.4500, 0.3000]])
# tensor([[0.9000, 3.0000, 0.1500, 0.0500, 0.7500, 0.3000]]) - lr_clus a bit fast?

# c=0.3/~.1, phi=1/3/5, lr_attn=.05/.15/.35, lr_nn=.05/.15/.25, lr_clus=.15/.45. lr_group .3/.9


# finegsearch 1 - dist**2 - all similar, with type 3 too  fast
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])

# finegsearch 2 - dist - v similar to original in plot
# w = torch.tensor([1/5, 1/5, 100/5, 1/5, 1/5])  #  pretty gd, 3 a little fast
# w = torch.tensor([1/5, 1/5, 500/5, 100/5, 1/5]) # also ok, all slower
# w = torch.tensor([10/5, 1/5, 1/5, 1/5, 1/5])  # total sse-->all faster, but dominates, nth changes if total sse w is high

# w = torch.tensor([10/5, 1/5, 1000/5, 100/5, 1/5])

# finegsearch 2 - dist**2. like above dist**2, all v similar, type 3 fast
# tensor([[0.4000, 1.5000, 0.4000, 0.0500, 0.4500, 0.9000]])
# tensor([[0.4000, 0.7500, 0.4000, 0.2000, 0.4500, 0.9000]])  # exactly same curve as aboev
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])
# w = torch.tensor([1/5, 1/5, 1000/5, 100/5, 1/5])


# for finegsearch dist these are ok i think
# # tensor([[0.3000, 0.7500, 0.9500, 0.3500, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 100/5, 1/5, 1/5, 1/5])   # same as before
# # tensor([[0.3000, 0.7500, 0.9500, 0.2500, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 1000/5, 1/5, 1/5, 1/5])  # slower, but more separation for 1-2 and 345-6

# w = torch.tensor([1/5, 1/5, 1/5, 1000/5, 1/5, 1/5])  # fiting 2-3 / 2-345 the same effect (as above)


# NEW - add diff between type 2-3 (3rd value)
# gsearch dist**2 - these are good patterns, with 3-5 equal, but a bit slow
# - maybe should do finegsearch from these params?
# # tensor([[0.7000, 1.0000, 0.1500, 0.6500, 0.7500, 0.1000]])
# w = torch.tensor([1/5, 1/5, 1000/5, 1/5, 300/5, 1/5])
# # tensor([[0.7000, 1.0000, 0.1500, 0.4500, 0.7500, 0.3000]])
# w = torch.tensor([1/5, 1/5, 1/5, 1000/5, 100/5, 1/5])  # sim to above - lr_group = .1/.3
# w = torch.tensor([1/5, 1/5, 0.01/5, 1000/5, 100/5, 1/5])  # same as above - maybe no need 2-3 diff?


# values to play around with:
# tensor([[0.3000, 0.7500, 0.9500, 0.3500, 0.4500, 0.9000]])  # finegsearrch dist - same as before type 3 fast
# tensor([[0.3000, 0.7500, 0.9500, 0.2500, 0.4500, 0.9000]])  # finegsearch dist, slightly slower
# tensor([[0.7000, 1.0000, 0.1500, 0.6500, 0.7500, 0.1000/0.3000]])  # gsearch dist**2 - slower, gd pattern


# ****
# these fingesearch might not be right - didn't scale lr_nn

# distsq2
# # tensor([[0.8000, 0.2500, 0.2500, 0.5000, 0.5500, 0.2000]])
# w = torch.tensor([1/5, 1/5, 1/5, 100/5, 1/5, 1/5])  # good pattern and separation, just not big enough sep
# # tensor([[0.4000, 0.5000, 0.5500, 0.3500, 0.6500, 0.1000]])
# w = torch.tensor([1/5, 1/5, 300/5, 200/5, 100/5, 1/5])  # slower but gd pattern
# # tensor([[0.4000, 0.5000, 0.5500, 0.3500, 0.7500, 0.2000]])
# w = torch.tensor([1/5, 1/5, 500/5, 200/5, 50/5, 1/5])  # similar to just above.. but all closer together, not as gd

# # dist1
# # - looks like the 1-2 and 2-345 diff is better than above
# # tensor([[0.4000, 0.7500, 0.9500, 0.1500, 0.5500, 0.8000]])
# w = torch.tensor([1/5, 1/5, 100/5, 500/5, 50/5, 1/5])  # pretty good. 3rd param, from 100-400 and 500-600 slightl diff - just lr_group .6 vs .8



# gsearch attn > 1 = looks great
# tensor([[0.4000, 7.0000, 2.7500, 0.0500, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5, 1/5])  # v gd
# tensor([[0.4000, 7.0000, 2.7500, 0.0500, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 100/5, 100/5])  # v sim

# with just diffs and assume all 0s
# tensor([[1.0000, 3.0000, 2.7500, 0.0500, 0.2500, 0.7000]])
w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])
# tensor([[0.4000, 7.0000, 2.7500, 0.0500, 0.4500, 0.9000]])
w = torch.tensor([1/5, 1/5, 1/5, 3/5, 1/5])  # this is like above, gd


# finegsearch attn
# with just diffs and assume all 0s - i think this is enough
# tensor([[0.4000, 5.0000, 2.7500, 0.1050, 0.4500, 0.9000]])
w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])  # looks gd already
# gd ones - weighting 3rd val
# tensor([[0.5000, 4.0000, 2.0000, 0.1050, 0.4500, 0.7000]])  # 3rd: 15/5 - ok but 2-345 a lil bit close
# tensor([[0.2000, 8.0000, 3.7500, 0.1450, 0.5500, 0.9000]])  # 20/5 - type 3 pops down a little
# tensor([[0.2000, 9.0000, 3.2500, 0.1250, 0.6500, 0.7000]])  # 25/5
# tensor([[0.2000, 8.0000, 3.2500, 0.1450, 0.6500, 0.7000]])  # 35/5. above this is same
# w = torch.tensor([1/5, 1/5, 20/5, 1/5, 1/5])  # probably 20 or 35/5
# w = torch.tensor([1/5, 1/5, 50/5, 10/5, 10/5])  # same as 20/5 above

# with nans - gd ones end up similar
# # tensor([[0.4000, 5.0000, 2.7500, 0.1050, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 20/5, 20/5, 1/5])
# # tensor([[0.2000, 8.0000, 2.2500, 0.1450, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 50/5, 20/5, 1/5])  # type 3 a lil slow
# tensor([[0.4000, 4.0000, 2.7500, 0.1450, 0.4500, 0.9000]])
# w = torch.tensor([1/5, 1/5, 50/5, 30/5, 1/5])  # type 2-345 closer

# nbanks

# just type 1 and 6 - flipped and should be faster across banks
# tensor([[6.0000e-01, 1.1250e+00, 1.6100e+00, 7.6000e-01, 3.0000e-01, 8.0000e-01,
#          1.8000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 8.0000e-01]])
w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])  # already OK

# similar, but shift up  bit - lines a little more different, type 6 bank 2-1 difference slightly more
# tensor([[5.0000e-01, 1.5000e+00, 2.0100e+00, 4.6000e-01, 3.0000e-01, 8.0000e-01,
#          1.8000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 8.0000e-01]])
# w = torch.tensor([1/5, 1/5, 500/5, 500/5, 1/5])

# final and works - include rulex types - should be faster in bank 2
# - standard OK except 345 and 6 too close
# tensor([[6.0000e-01, 1.1250e+00, 8.1000e-01, 6.1000e-01, 3.0000e-01, 8.0000e-01,
#          2.0000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 8.0000e-01]])
w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])

# a bit more separated than above
# tensor([[5.0000e-01, 1.1250e+00, 2.0100e+00, 7.6000e-01, 3.0000e-01, 8.0000e-01,
         # 1.8000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 8.0000e-01]])
# w = torch.tensor([1/5, 1/5, 1/5, 150/5, 1/5])  # from 150


# finegsearch - testing without 3 sets
# tensor([[6.0000e-01, 1.1250e+00, 1.5500e+00, 7.0000e-01, 3.0000e-01, 8.0000e-01,
         # 1.7000e+00, 2.5000e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
w = torch.tensor([1/5, 1/5, 1/5, 150/5, 1/5])
# tensor([[6.0000e-01, 1.0000e+00, 1.5500e+00, 7.0000e-01, 3.0000e-01, 8.0000e-01,
#          1.7000e+00, 2.2500e+00, 1.0000e-03, 1.0000e-02, 3.0000e-01, 5.0000e-01]])
w = torch.tensor([1/5, 1/5, 350/5, 350/5, 1/5]) # shifted upm maybe a bit better - need get exact good numbers though. this is what's in the figure now

# with type II as 2nd slower in bank 2
# w = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5])
# w = torch.tensor([1/5, 1000/5, 1/5, 1/5, 1/5]) # no diff
# w = torch.tensor([1/5, 1/5, 100/5, 100/5, 1/5])  # meh


w = w / w.sum()
sses_w = sse * w[0] + torch.sum(sse_diff * w[1:], axis=1)
ind_sse_w = sses_w == sses_w[ptn_criteria].min()

if len(torch.nonzero(ind_sse_w)) > 1:
    ind_sse_w[torch.nonzero(ind_sse_w)[1]] = 0

# # RANK
# # ranked_nll = [sorted(nlls).index(i) for i in nlls]
# from scipy.stats import rankdata

# # note, scipy.stats.rankdata starts from 1
# ranked_nll = rankdata(nlls, method='min')
# ranked_sse = rankdata(sse, method='min')
# ranked_sse_diff = rankdata(sse_diff, method='min')
# # ignore
# ranked_nll[~ptn_criteria] = 10**6
# ranked_sse[~ptn_criteria] = 10**6
# ranked_sse_diff[~ptn_criteria] = 10**6

# # since there are ties, get the first n values
# irank = 0 # start from 0

# ranks = np.sort(ranked_nll)
# ind_nll = ranked_nll == ranks[irank]
# ranks = np.sort(ranked_sse)
# ind_sse = ranked_sse == ranks[irank]
# ranks = np.sort(ranked_sse_diff)
# ind_sse_diff = ranked_sse_diff == ranks[irank]
# # use ranks
# ranked_sse_w = rankdata(sses_w, method='min')
# ranked_sse_w[~ptn_criteria] = 10**6  # ignore
# ranks = np.sort(ranked_sse_w)
# ind_sse_w = ranked_sse_w == ranks[irank]


# c, phi, lr_attn, lr_nn, lr_clusters, lr_clusters_group
# print(param_sets[ind_nll])
# print(param_sets[ind_sse])
# print(param_sets[ind_sse_diff])
# print(param_sets[ind_sse_diff1])
# print(param_sets[ind_sse_diff2])

print(param_sets[ind_sse_w])

# plt.plot(nlls[ptn_criteria])
# # plt.ylim([88, 97])
# plt.show()
# plt.plot(sse[ptn_criteria])
# # plt.ylim([0, 9])
# plt.show()
# plt.plot(sses_w[ptn_criteria])
# plt.show()

# select which to use
ind = ind_sse_w

# ind = ind_nll.clone()

# if more than 1 match, remove one
# ind[torch.nonzero(ind_nll)[0]] = True
# ind[torch.nonzero(ind_nll)[1]] = False

# more criteria
# - maybe faster type I / slower type VI
# - types III-V need to be more similar (e.g. within some range)
# - type I, II and the III-V's need to be larger differences

# % plot

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




# check best params to do a finer gridsearch

# new - squared differences for matching the differences
# dist
# - basically all w's give the same: 345 together, but also close to 6
# tensor([[2.0000, 1.0000, 0.8500, 0.1500, 0.2500, 0.9000]])
# - but with all separate, this gives more varied curves

# dist**2 with sqdiff for fitting
# - also not a lot of range. 3 a bit slow
# - with all separate

# param_sets = torch.tensor(list(it.product(*ranges)))

print(len(param_sets))


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

# fig, ax = plt.subplots(1, 1)
# ax.plot(pts_banks[ind, :, 0].T.squeeze())
# ax.tick_params(axis='x', labelsize=fntsiz-3)
# ax.tick_params(axis='y', labelsize=fntsiz-3)
# ax.set_ylim(ylims)
# ax.set_title('Module 1', fontsize=fntsiz)
# ax.set_xlabel('Learning Block', fontsize=fntsiz)
# ax.set_ylabel('Probability of Error', fontsize=fntsiz)
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.plot(pts_banks[ind, :, 1].T.squeeze())
# ax.tick_params(axis='x', labelsize=fntsiz-3)
# ax.tick_params(axis='y', labelsize=fntsiz-3)
# ax.set_ylim(ylims)
# ax.set_title('Module 2', fontsize=fntsiz)
# ax.set_xlabel('Learning Block', fontsize=fntsiz)
# ax.set_ylabel('Probability of Error', fontsize=fntsiz)
# plt.tight_layout()
# plt.show()

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
