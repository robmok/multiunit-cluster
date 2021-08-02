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
resdir = os.path.join(maindir, 'muc-shj-gridsearch/gsearch_k0.01_500units')

k = 0.01
n_units = 500
iset = 0

fn = os.path.join(resdir,
                  'shj_gsearch_k{}_{}units_set{}.pkl'.format(k, n_units, iset))

# load - list: [nlls, pt_all, rec_all, seeds_all]
open_file = open(fn, "rb")
loaded_list = pickle.load(open_file)
open_file.close()


pts = torch.stack(loaded_list[1])

iparam = 0
plt.plot(pts[iparam].T)

# for iparam in range(10):
#     plt.plot(pts[iparam].T)
#     plt.show()