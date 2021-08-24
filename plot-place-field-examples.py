#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:26:09 2021

Example place fields

- currently just copy and pasted stuff from sim
- can make it 'simpler' - just get activations of the gaussian...

@author: robert.mok
"""

saveplots = True

path_test = torch.tensor(
    np.around(np.random.rand(20000, 2), decimals=3),
    dtype=torch.float32)

loc = [.25, .25]
loc = [.5, .25]
loc = [.75, .75]
loc = [.25, .75]
cov = torch.cholesky(torch.eye(2) * .01)  # .05, .01
mvn1 = torch.distributions.MultivariateNormal(torch.tensor(loc),
                                              scale_tril=cov)
nbins = 40
act_test = []
for itrial in range(len(path_test)):
    act = torch.exp(mvn1.log_prob(path_test[itrial].detach()))
    act_test.append(act.detach())
act_map = _compute_activation_map(
    path_test, torch.tensor(act_test), nbins, statistic='sum')
norm_mat = normalise_act_map(nbins, act_map.binnumber)

ind = np.nonzero(norm_mat)
act_map_norm = act_map.statistic.copy()
act_map_norm[ind] = act_map_norm[ind] / norm_mat[ind]


fig, ax = plt.subplots()
ax.imshow(act_map_norm)
# ax.set_title('k = {}'.format(k))
ax.set_xticks([])
ax.set_yticks([])
if saveplots:
    figname = os.path.join(
        figdir, 'actmaps/'
        'place_field_example_loc{:.2f}-{:.2f}_cov{:.3f}'.format(loc[0], loc[1],
                                                                cov[0, 0]))
    # plt.savefig(figname, dpi=100)
    plt.savefig(figname + '.pdf')
plt.show()
