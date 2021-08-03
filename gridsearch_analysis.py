#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:57:19 2021

@author: robert.mok
"""

import os
# import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

maindir = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/'

k = 0.01
n_units = 500

resdir = os.path.join(maindir, 'muc-shj-gridsearch/gsearch_k{}_{}units'.format(
    k, n_units))

pts = []
nlls = []
recs = []
seeds = []

for iset in range(250):
    fn = os.path.join(
        resdir,
        'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

    # load - list: [nlls, pt_all, rec_all, seeds_all]
    open_file = open(fn, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    nlls.extend(loaded_list[0])
    pts.extend(loaded_list[1])
    # recs.extend(loaded_list[2])
    # seeds.extend(loaded_list[3])

pts = torch.stack(pts).T
nlls = torch.stack(nlls)
# recs = torch.stack(recs)
# seeds = torch.stack(seeds)

# %%

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

# qualitiative
# - pattern - 1, 2, 3-4-5, 6 (where 3, 4, 5 can be in any order for now). all
# points have to be faster (for now - maybe do 80% points if problem?)
# -

iparam = 0



# match threshold
# - 1 if match fully. can allow some error to be safe, eg ~.9
# - note depending on the comparison, num total is diff (so prop is diff)
match_thresh = .95

# pattern, meet criteria?
sse = torch.zeros(len(pts))
ptn_criteria_1 = torch.zeros(len(pts), dtype=torch.bool)
for iparam in range(len(pts)):

    # compute sse
    sse[iparam] = torch.sum(torch.square(pts[iparam].T.flatten() - shj.flatten()))

    # pattern
    ptn = pts[iparam][0] < pts[iparam][1:]  # type I fastest
    ptn_c1 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    ptn = pts[iparam][1] < pts[iparam][2:]  # type II 2nd
    ptn_c2 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh
    ptn = pts[iparam][5] > pts[iparam][:5]  # type VI slowest
    ptn_c3 = torch.sum(ptn) / torch.numel(ptn) >= match_thresh

    ptn_criteria_1[iparam] = ptn_c1 & ptn_c2 & ptn_c3


# criteria 1 already reduces to <12% of the params

# sse and nll very similar, though LOWEST value differ. interesting
plt.plot(nlls[ptn_criteria_1])
plt.show()
plt.plot(sse[ptn_criteria_1])
plt.show()

ind_nll = nlls == nlls[ptn_criteria_1].min()
ind_sse = sse == sse[ptn_criteria_1].min()

# more criteria
# - maybe faster type I / slower type VI
# - types III-V need to be more similar (e.g. within some range)
# - type I, II and the III-V's need to be larger differences





# %% plot

fig, ax = plt.subplots(2, 1)
ax[0].plot(shj.T)
ax[0].set_ylim([0., .55])
ax[0].set_aspect(17)
ax[1].plot(pts[ind_nll].T.squeeze())
ax[1].set_ylim([0., .55])
ax[1].legend(('1', '2', '3', '4', '5', '6'), fontsize=7)
ax[1].set_aspect(17)
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(shj.T, 'k')
ax.plot(pts[ind_nll].T.squeeze(), 'o-')
ax.set_ylim([0., .55])
ax.legend(('1', '2', '3', '4', '5', '6', '1', '2', '3', '4', '5', '6'), fontsize=7)