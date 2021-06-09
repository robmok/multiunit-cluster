#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:11:29 2021

Noise distributions to sample from

@author: robert.mok
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# unit positions - can be pos/neg (imprecision of placement)
mu = 0
sd = .05
x = np.linspace(norm(loc=mu, scale=sd).ppf(0.0001),
                norm(loc=mu, scale=sd).ppf(0.9999), 100)
rv = norm(loc=mu, scale=sd)
plt.plot(x, rv.pdf(x), 'k-', lw=2)

# update
# for placement, more is fine, but update might need to scale to clus lr
# - updates are: lr~.1-.2, dist~.5 =  ~.05-.1.
# - noise should be a proportion of this
mu = 0
sd = .01
x = np.linspace(norm(loc=mu, scale=sd).ppf(0.0001),
                norm(loc=mu, scale=sd).ppf(0.9999), 100)
rv = norm(loc=mu, scale=sd)
plt.plot(x, rv.pdf(x), 'k-', lw=2)

# unit activations - just positive ('adding noise' to acts)
mu = .5
sd = .1
x = np.linspace(norm(loc=mu, scale=sd).ppf(0.0001),
                norm(loc=mu, scale=sd).ppf(0.9999), 100)
rv = norm(loc=mu, scale=sd)
plt.plot(x, rv.pdf(x), 'k-', lw=2)

# sampling
r = norm.rvs(loc=0, scale=.1, size=5000)
plt.hist(r)
plt.show()

r = norm.rvs(loc=0.5, scale=.1, size=5000)
plt.hist(r)
plt.show()

# %% examples

# pos
mu = 0
sd = .1  # .2 gets 0 to .5 at the tails
pos = np.array([0, 0, 1])  # stim position
# print(pos + norm.rvs(loc=mu, scale=sd, size=3))
plt.hist(pos[np.newaxis] + norm.rvs(loc=mu, scale=sd, size=(1000, 3)))
plt.show()

# update
mu = 0
sd = .01
upd = np.array([-.1, -.05, 0, 0.05, .1])
plt.hist(upd[np.newaxis] + norm.rvs(loc=mu, scale=sd, size=(1000, 5)), bins=25)
plt.show()

# act
mu = .5
sd = .15  # .1 ok but might be too little, .15 gd, .2 getting too much?
act = np.array([.8, 0.5, 0.])
# print(act + norm.rvs(loc=mu, scale=sd, size=3))

plt.hist(act[np.newaxis] + norm.rvs(loc=mu, scale=sd, size=(1000, 3)))
plt.show()
