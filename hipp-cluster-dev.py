#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:07:51 2020

Hippocampal (lower level) clustering model

Learning rule (double update):
- winning units all move toward the current stimulus (Kohonen learning rule)
and then a second update (different learning rate parameter) has them all move
toward their centroid (i.e., toward each other).

Basic stuff:
act = torch.exp(-c * dist**2)
dist = attention-weighted euclidean distance [have the option for city-block]

Determining the winning units, and making a decision
- top k% (a parameter) of strongest activated units are winners
- if not enough units active at all (e.g. early trials), then winners < k%
- only winners have nonzero output
- evidence = w * outputs (w are association weights to category decision)
- pr(decicions) = exp(d*evidence) / sum(exp(d*evidnece)) [summing over possible decisions / categories]

Recruiting clusters
- new units to be recruited? when not enough units with nonzero output (e.g.
1st trial) and when winning units signal the wrong category. Winning units
mis-predict when their largest weight is not toward the correct
- those winners -> non-winners (set output to 0) and replaced with another unit
- the replacement (i.e., recruited) winners are unconnected units (i.e., all
wij are 0) that have the greatest act.  Set their new position pos equal to x.

Learning rule updates
- cluster update 1: winners update towards stimulus ∆posi = lr_pos · (x − posi)
- cluster update 2: winners update towards winner centroid. ∆posi = lr_group · (group_mean_pos − posi),
- attention weights:
    - either by gradient descent (pytorch)
    - or: gradient ascent on the sum of the outputs of the winning units (check brad's email reply to my question on this)
- association weights

Parameters summary:
- 4 learning rates (lr_pos, lr_group, lr_attn, lr_assoc_W)
- dispersion c parameter
- softmax decision d. i think this is phi (here and alcove)
- k percent winners
- pattern separation (default=1) - not sure where this is? CHECK
- unsupervised recruitment parameter (only for unsupervised learning)


--

Gradient ascent from Brad:
" it’s completely focused on the winners just like SUSTAIN’s attention learning
rule, but less hackish. It would be the derivative that relates changes in
unit’sactivation to its attention weight with the sign positive instead of
flipped negatively for gradient descent. It’s how to change the attention
weights to maximise these units’ sum of activity. Instead of an error term,
you have the sum of activity of the winning units and the parameter that
changes is the attention weights. In that sense, it’s just like the normal
gradient descent except that the measure is maximised, not minimised. "

@author: robert.mok
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools as it


class MultiUnitCluster(nn.Module):
    def __init__(self, n_units, n_dims, attn_type, k, params=None):
        super(MultiUnitCluster, self).__init__()
        self.attn_type = attn_type
        self.n_units = n_units
        self.n_dims = n_dims
        self.softmax = nn.Softmax(dim=0)
        # history
        self.attn_trace = []
        self.units_pos_trace = []
        self.units_act_trace = []
        self.recruit_units_trl = []
        self.fc1_w_trace = []
        self.fc1_act_trace = []

        # checking stuff
        self.winners_trace = []
        self.dist_trace = []
        self.act_trace = []

        # free params
        # - can estimate all, but probably don't need to include some (e.g. r)
        if params:
            self.params = params
        else:
            self.params = {
                'r': 1,  # 1=city-block, 2=euclid
                'c': 2,  # node specificity
                'p': 1,  # alcove: p=1 exp, p=2 gauss
                'phi': 1,  # response parameter, non-negative
                'lr_attn': .25,
                'lr_nn': .25,
                'lr_clusters': .15,
                'lr_clusters_group': .95,
                'k': k
                }

        # units
        self.units_pos = torch.zeros([n_units, n_dims], dtype=torch.float)

        # randomly scatter
        self.units_pos = torch.rand([n_units, n_dims], dtype=torch.float)

        # # cluster positions as trainable parameters
        # self.clusters = torch.nn.Parameter(
        #     torch.zeros([n_units, n_dims], dtype=torch.float))

        # attention weights - 'dimensional' = ndims / 'unit' = clusters x ndim
        if self.attn_type == 'dimensional':
            self.attn = (torch.nn.Parameter(
                torch.ones(n_dims, dtype=torch.float) * .33))
            # normalize attn to 1, in case not set correctly above
            self.attn.data = (
                        self.attn.data / torch.sum(self.attn.data))
        elif self.attn_type == 'unit':
            self.attn = (
                torch.nn.Parameter(torch.ones([n_units, n_dims],
                                              dtype=torch.float) * .33))
            # normalize attn to 1, in case not set correctly above
            self.attn.data = (
                self.attn.data /
                torch.sum(self.attn.data, dim=1, keepdim=True))

        # network to learn association weights for classification
        n_classes = 2  # n_outputs
        self.fc1 = nn.Linear(n_units, n_classes, bias=False)
        self.fc1.weight = torch.nn.Parameter(torch.zeros([n_classes, n_units]))

        # mask for NN
        self.mask = torch.zeros([n_classes, n_units], dtype=torch.bool)

        # mask for updating attention weights based on winning units
        # - might not need this. check with unit-based attention
        # - winning_units is like active_units before, but winning on that
        # trial, since active is define by connection weight ~=0
        # mask for winning clusters
        self.winning_units = torch.zeros(n_units, dtype=torch.bool)
        # self.mask = torch.zeros([n_classes, n_units], dtype=torch.bool)
        # # do i need this? - i think no, just to make starting weights 0
        # with torch.no_grad():
        #     self.fc1.weight.mul_(self.mask)

    def forward(self, x):
        # compute activations of clusters here. stim x clusterpos x attn

        # distance measure. *attn works for both dimensional or unit-based
        dim_dist = abs(x - self.units_pos)
        dist = _compute_dist(dim_dist, self.attn, self.params['r'])

        # compute attention-weighted dist & activation (based on similarity)
        act = _compute_act(dist, self.params['c'], self.params['p'])

        norm_units = False
        if norm_units:
            # beta = self.params['beta']
            beta = 1
            act.data[self.winning_units] = (
                (act.data[self.winning_units]**beta) /
                (torch.sum(act.data[self.winning_units]**beta)))

        units_output = act * self.winning_units

        # save cluster positions and activations
        self.units_pos_trace.append(self.units_pos.detach().clone())
        self.units_act_trace.append(units_output.detach().clone())

        # save attn weights
        self.attn_trace.append(self.attn.detach().clone())

        # association weights / NN
        out = self.fc1(units_output)
        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out.detach().clone())

        # convert to response probability
        pr = self.softmax(self.params['phi'] * out)

        return out, pr


def train(model, inputs, labels, n_epochs, loss_type='cross_entropy',
          shuffle=False):

    if loss_type == 'humble_teacher':
        criterion = humble_teacher
    elif loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
    p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
    params = [p_fc1, p_attn]

    # model.params['lr_clusters'], model.params['lr_clusters_group'],

    optimizer = optim.SGD(params, lr=model.params['lr_nn'])  # , momentum=0.)

    # save accuracy
    itrl = 0
    trial_acc = torch.zeros(len(inputs) * n_epochs)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros(len(inputs) * n_epochs)
    epoch_ptarget = torch.zeros(n_epochs)

    model.train()
    for epoch in range(n_epochs):
        if shuffle:
            shuffle_ind = torch.randperm(len(inputs))
            inputs_ = inputs[shuffle_ind]
            labels_ = labels[shuffle_ind]
        else:
            inputs_ = inputs
            labels_ = labels
        for x, target in zip(inputs_, labels_):
            
            # TMP - testing
            x=inputs_[np.mod(itrl-8, 8)]
            target=labels_[np.mod(itrl-8, 8)]
            
            # find winners
            # first: only connected units (assoc ws ~0) can be winners
            # - any weight > 0 = connected/active unit (so sum over out dim)
            active_ws = torch.sum(abs(model.fc1.weight) > 0, axis=0,
                                  dtype=torch.bool)

            # find units with largest activation that are connected
            dim_dist = abs(x - model.units_pos)
            dist = _compute_dist(dim_dist, model.attn, model.params['r'])
            act = _compute_act(dist, model.params['c'], model.params['p'])
            act[~active_ws] = 0  # not connected, no act
            # _, ind_dist = torch.sort(act)
            # get top k winners
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[win_ind] == 0):
                win_ind = win_ind[act[win_ind] != 0]

            if itrl > 0:
                model.dist_trace.append(dist[win_ind][0].detach().clone())
                model.act_trace.append(act[win_ind][0].detach().clone())

            # define winner mask
            winners_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
            winners_mask[:, win_ind] = True
            # this goes into forward. if ~active, no out
            model.winning_units = torch.zeros(n_units, dtype=torch.bool)
            model.winning_units[win_ind] = True

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            # zero out gradient for masked connections
            with torch.no_grad():
                model.fc1.weight.grad.mul_(winners_mask)
                if model.attn_type == 'unit':  # mask other clusters' attn
                    model.attn.grad.mul_(winners_mask[0].unsqueeze(0).T)

            # update model - if inc/recruit a cluster, don't update here
            # if incorrect, recruit
            if ((not torch.argmax(out.data) == target) or
               (torch.all(out.data == 0))):  # if incorrect
                recruit = True
            else:
                recruit = False

            # if not recruit, update model
            if recruit:
                pass
            else:
                optimizer.step()
                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                if model.attn_type == 'dimensional':
                    model.attn.data = (
                        model.attn.data / torch.sum(model.attn.data))
                elif model.attn_type == 'unit':
                    model.attn.data = (
                        model.attn.data /
                        torch.sum(model.attn.data, dim=1, keepdim=True)
                        )

                # update units - double update rule
                # - step 1 - winners update towards input
                update = (
                    (x - model.units_pos[win_ind]) *
                    model.params['lr_clusters']
                    )
                model.units_pos[win_ind] += update

                # - step 2 - winners update towards self
                winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
                update = (
                    (winner_mean - model.units_pos[win_ind]) *
                    model.params['lr_clusters_group'])
                model.units_pos[win_ind] += update

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out.data) == target
            trial_ptarget[itrl] = pr[target]

            # Recruit cluster, and update model
            # - here, recruitment means get a subset of units with nn_weights=0, place it at curr stim

            if recruit:
                # select random k units
                # inactive_ind = torch.nonzero(active_ws == False)
                # rand_k_units = (
                #     torch.randint(len(inactive_ind),
                #                   (int(model.n_units * model.params['k']), ))
                #     )
                # recruit_ind = inactive_ind[rand_k_units]

                # 1st trial: select closest k inactive units
                if torch.all(~active_ws):  # no active weights / 1st trial
                    act = _compute_act(
                        dist, model.params['c'], model.params['p'])
                    _, recruit_ind = (
                        torch.topk(act,
                                   int(model.n_units * model.params['k'])))
                    # since topk takes top even if all 0s, remove the 0 acts
                    if torch.any(act[recruit_ind] == 0):
                        recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit and REPLACE k units that mispredicted
                # - TODO - this does not work for rulex, since it replaces all
                # the units, since all mispredict.
                # - I got it - don't 'replace', just set the act to zero
                # and recruit n units. normall this will just be k units.
                # - *BUT* what is set to 0? by having new units, those will be on
                # the stim, but the old units might also be.. i guess since the
                # weight's connected to the new units should point in the right
                # directions since i update them below, those should be the winners?
                else:
                    mispred_units = torch.argmax(
                        model.fc1.weight[:, win_ind].detach(), dim=0) != target

                    # select closest n_mispredicted inactive units
                    n_mispred_units = len(mispred_units)
                    act = _compute_act(
                        dist, model.params['c'], model.params['p'])
                    act[active_ws] = 0  # REMOVE active units
                    _, recruit_ind = (
                        torch.topk(act, n_mispred_units))
                    # since topk takes top even if all 0s, remove the 0 acts
                    if torch.any(act[recruit_ind] == 0):
                        recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit n_mispredicted units
                active_ws[recruit_ind] = True  # set ws to active
                model.winning_units = torch.zeros(n_units, dtype=torch.bool)
                model.winning_units[recruit_ind] = True
                model.units_pos[recruit_ind] = x  # place at curr stim
                # model.mask[:, active_ws] = True  # new clus weights
                model.recruit_units_trl.append(itrl)

                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                with torch.no_grad():
                    # model.fc1.weight.grad.mul_(model.mask)  # win_mask enuf?
                    win_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
                    win_mask[:, active_ws] = True  # update new clus
                    model.fc1.weight.grad.mul_(win_mask)
                    if model.attn_type == 'unit':
                        model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)
                optimizer.step()
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                if model.attn_type == 'dimensional':
                    model.attn.data = (
                        model.attn.data / torch.sum(model.attn.data))
                elif model.attn_type == 'unit':
                    model.attn.data = (
                        model.attn.data /
                        torch.sum(model.attn.data, dim=1, keepdim=True)
                        )

                # update units - double update rule
                # - no need this, since already placed at the stim?
                
            # tmp
            model.winners_trace.append(model.units_pos[model.winning_units][0])

            itrl += 1

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        epoch_ptarget[epoch] = trial_ptarget[itrl-len(inputs):itrl].mean()

    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget


def train_unsupervised(model, inputs, n_epochs):

    for epoch in range(n_epochs):
        shuffle_ind = torch.randperm(len(inputs))
        inputs_ = inputs[shuffle_ind]
        # inputs_ = inputs
        for x in inputs_:
            # find winners
            # find units with largest activation - all connected
            dim_dist = abs(x - model.units_pos)
            dist = _compute_dist(dim_dist, model.attn, model.params['r'])
            act = _compute_act(dist, model.params['c'], model.params['p'])

            # _, ind_dist = torch.sort(act)
            # get top k winners
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[win_ind] == 0):
                win_ind = win_ind[act[win_ind] != 0]

            # define winner mask
            winners_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
            winners_mask[:, win_ind] = True

            # learn
            # update units - double update rule
            # - step 1 - winners update towards input
            update = (
                (x - model.units_pos[win_ind]) * model.params['lr_clusters'])
            model.units_pos[win_ind] += update

            # - step 2 - winners update towards self
            winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
            update = (
                (winner_mean - model.units_pos[win_ind]) *
                model.params['lr_clusters_group'])
            model.units_pos[win_ind] += update

            # store positions over time
            model.units_pos_trace.append(model.units_pos.detach().clone())


def _compute_dist(dim_dist, attn_w, r):
    return torch.sum((attn_w * dim_dist)**r, axis=1)**(1/r)


def _compute_act(dist, c, p):
    """ c = 1  # ALCOVE - specificity of the node - free param
        p = 2  # p=1 exp, p=2 gauss
    """
    return torch.exp(-c * (dist**p))


# loss functions
def humble_teacher(output, target, n_classes=2):
    '''
    If multiple_tasks or output classes>2, need specify n_classes
    (unlike cross_entropy - so I had an if statement for loss=criterion...)
    '''
    # # 1 output, sustain
    # if (target == 1 and output >= 1) or (target == 0 and output <= 0):
    #     error = 0
    # else:
    #     error = target - output  # check

    # 2+ outputs
    output = output.squeeze()  # unsqueeze needed for cross-entropy
    error = torch.zeros(n_classes)
    target_vec = torch.zeros(n_classes)
    target_vec[target] = 1
    for i in range(n_classes):
        if ((target_vec[i] == 1 and output[i] >= 1) or
                (target_vec[i] == 0 and output[i] <= -1)):
            error[i] = output[i] - output[i]  # 0. do this to keep the grad_fn
        else:
            # if in category K, t=1, else t=-1
            if target_vec[i] == 1:
                t = 1
            elif target_vec[i] == 0:
                t = -1
            error[i] = .5 * ((t - output[i])**2)
    return torch.sum(error)


# %%

"""

inputs, n_units, n_dims, loss, n_epochs

k - proportion winners

"""

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

                # type 1 continuous - 2D
                [[.75,   0,   .75,   0.,  0],
                 [.5,   .25,  .5,   .25,  0],
                 [.25,  .5,   .25,  .5,   1],
                 [0.,   .75,   0.,  .75,  1],
                 [.75,   0.,   0.,  .75,  0],
                 [.5,   .25,  .25,  .5,   0],
                 [.25,  .5,   .5,   .25,  1],
                 [0.,   .75,  .75,  .0,   1]],

                # type 1 continuous - 3D
                [[.75,   0,   .75,   0., .75,  0,  0],
                 [.5,   .25,  .5,   .25, .25, .5,  0],
                 [.25,  .5,   .25,  .5,  .5,  .25, 1],
                 [0.,   .75,   0.,  .75,  0., .75, 1],
                 [.75,   0.,   0.,  .75,  0., .75, 0],
                 [.5,   .25,  .25,  .5,  .5,  .25, 0],
                 [.25,  .5,   .5,   .25, .25, .5,  1],
                 [0.,   .75,  .75,  .0,  .75,  0., 1]],
                ]

# set problem
problem = 4
stim = six_problems[problem]
stim = torch.tensor(stim, dtype=torch.float)
inputs = stim[:, 0:-1]
output = stim[:, -1].long()  # integer

# # continuous
# mu1 = [-.5, .25]
# var1 = [.0185, .065]
# cov1 = -.005
# mu2 = [-.25, -.6]
# var2 = [.0125, .005]
# cov2 = .005

# # same/similar on first dim - attn not learning the right one...?
# mu1 = [-.5, .25]
# var1 = [.0185, .065]
# cov1 = -.005
# mu2 = [-.5, -.7]
# var2 = [.015, .005]
# cov2 = .005

# # simple diagonal covariance
# mu1 = [-.5, .25]
# var1 = [.02, .02]
# cov1 = 0
# mu2 = [-.25, -.6]
# var2 = [.02, .02]
# cov2 = 0

# # mu1 = [-.5, .25]
# # var1 = [.0185, .065]
# # cov1 = 0
# # mu2 = [-.25, -.6]
# # var2 = [.0125, .005]
# # cov2 = 0

# npoints = 100
# x1 = np.random.multivariate_normal(
#     [mu1[0], mu1[1]], [[var1[0], cov1], [cov1, var1[1]]], npoints)
# x2 = np.random.multivariate_normal(
#     [mu2[0], mu2[1]], [[var2[0], cov2], [cov2, var2[1]]], npoints)

# # x1 = np.append(x1, np.zeros([npoints, 1]), axis=1)
# # x2 = np.append(x2, np.zeros([npoints, 1]), axis=1)

# inputs = torch.cat([torch.tensor(x1, dtype=torch.float32),
#                     torch.tensor(x2, dtype=torch.float32)])
# output = torch.cat([torch.zeros(npoints, dtype=torch.long),
#                     torch.ones(npoints, dtype=torch.long)])

# model details
attn_type = 'dimensional'  # dimensional, unit (n_dims x nclusters)
n_units = 1000
n_dims = inputs.shape[1]
# nn_sizes = [clus_layer_width, 2]  # only association weights at the end
loss_type = 'cross_entropy'
# c_recruit = 'feedback'  # feedback or loss_thresh

# top k%. so .05 = top 5%
k = .05

# spatial / unsupervised

# looks like k is key for number of virtual clusters that come up. smaller k = more; larger k = fewer clusters 
# lr_group has to be large-ish, else virtual clusters don't form (scattered).
# lr_group has to be > lr_clusters, else virtual cluster don't form. but not too high else clusters go toward centre

# - i think the learning rates might lead to more/less grid like patterns - check which matters more (can use banino's grid code)
# - need reduction of lr over time?

# SHJ
# - do I  want to save trace for both clus_pos upadtes? now just saving at the end of both updates


# normalization of units
# c=3, beta=1, lr_attn = .001 then looks OK. faster lr_attn screws up
# - n_units = 1000 k=.01 then 6. k=.05 then 8 (c=3/6 same - quick check seems c doesn't matter)

# To check
# - one thing i see from plotting over time is that clusters change sometimes change across virtual clusters. need lower lr?
# looks like less later on though. maybe ok?

# trials, etc.
n_epochs = 100

# attn
# - w attn ws starting at .5: lr_attn.005 for 6 clus
# - w attn w = .33 - not as good?
# - w attn w = .1  - lr_attn = .05, c=6. type V - bump in pr, looks like attn weights all go up, THEN irr go down

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': 10,  # node specificity - 6. hmm, if start attn at .33, type V needs c=12 for 6? act now ok, lr_nn = .05
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 1,  # response parameter, non-negative
    'lr_attn': .005,  # .005 / .05 / .001
    'lr_nn': .1,  # .1. .01 actually better, c=6. cont - .15
    'lr_clusters': .25,  # .25
    'lr_clusters_group': .95,  # .95
    'k': k
    }

# # continuous
# params = {
#     'r': 1,
#     'c': 10,
#     'p': 1,
#     'phi': 1,
#     'lr_attn': .025,  # cont+unit- .25 - needs to be fast..?
#     'lr_nn': .015,  # cont - .15
#     'lr_clusters': .125,
#     'lr_clusters_group': .95,
#     'k': k
#     }

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=True)

print(epoch_acc)
print(epoch_ptarget)
plt.plot(1 - epoch_ptarget.detach())
plt.show()

active_ws = torch.sum(abs(model.fc1.weight) > 0, axis=0, dtype=torch.bool)
# print(np.around(model.units_pos.detach().numpy()[active_ws], decimals=2))
print(np.unique(np.around(model.units_pos.detach().numpy()[active_ws], decimals=2), axis=0))
# print(np.unique(np.around(model.attn.detach().numpy()[active_ws], decimals=2), axis=0))
# print(model.attn)

print(len(model.recruit_units_trl))

# %% unsupervised
n_dims = 2
n_epochs = 1
n_trials = 2000
attn_type = 'dimensional'

inputs = torch.rand([n_trials, n_dims], dtype=torch.float)
n_units = 1000
k = .1

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': 2,  # node specificity 2/3 for cluster/wta/exem, w p=1 better
    'p': 1,  # alcove: p=1 exp, p=2 gauss
    'phi': 1,  # response parameter, non-negative
    'lr_attn': .25,
    'lr_nn': .25,
    'lr_clusters': .15,
    'lr_clusters_group': .95,
    'k': k
    }

# model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)
# train_unsupervised(model, inputs, n_epochs)

# results = torch.stack(model.units_pos_trace, dim=0)
# plt.scatter(results[-1, :, 0], results[-1, :, 1])
# plt.xlim([0, 1])
# plt.ylim([0, 1])    
# plt.show()


# run for different learning rates for lr_clusters and lr_group
# lr_clusters = torch.linspace(.001, .5, 10)
# lr_group = torch.linspace(.1, 2, 10)

lr_clusters = torch.arange(.001, .5, .05)
lr_group = torch.arange(.1, 2, .2)

results = torch.zeros(n_units, n_dims, len(lr_clusters), len(lr_group))
for i, j in it.product(range(len(lr_clusters)), range(len(lr_group))):
    params['lr_clusters'] = lr_clusters[i]
    params['lr_clusters_group'] = lr_group[j]
    model = MultiUnitCluster(n_units, n_dims, attn_type, k, params)
    train_unsupervised(model, inputs, n_epochs)
    results[:, :, i, j] = torch.stack(model.units_pos_trace, dim=0)[-1]


# fig, ax = plt.subplots(len(lr_clusters), len(lr_group))
# for i, j in it.product(range(len(lr_clusters)), range(len(lr_group))):
#     ax[i, j].scatter(results[:, 0, i, j], results[:, 1, i, j], s=.005)
#     ax[i, j].set_xlim([0, 1])
#     ax[i, j].set_ylim([0, 1])


wd = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/multiunit-cluster_figs'

lr = lr_group[3]
j = torch.nonzero(lr_group == lr)
for i in range(len(lr_clusters)):
    plt.scatter(results[:, 0, i, j], results[:, 1, i, j])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

    # figname = os.path.join(wd,
    #                         'hipp_cluster_across_lrclus' +
    #                         str(round(lr_clusters[i].tolist(), 3)) +
    #                         '_lrgroup' + str(round(lr.tolist(), 3)) + '.png')
    # plt.savefig(figname)
    # plt.show()


lr = lr_clusters[5]  # >.1 [3/4/5]
i = torch.nonzero(lr_clusters == lr)
for j in range(len(lr_group)):
    plt.scatter(results[:, 0, i, j], results[:, 1, i, j])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

    # figname = os.path.join(wd,
    #                        'hipp_cluster_across_lrgroup' +
    #                        str(round(lr_group[j].tolist(), 3)) +
    #                        '_lrclus' +
    #                        str(round(lr.tolist(), 3)) + '.png')
    # plt.savefig(figname)
    # plt.show()



# %% plot unsupervised

results = torch.stack(model.units_pos_trace, dim=0)

# # group
# plt.scatter(results[-1, :, 0], results[-1, :, 1])
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.show()

# over time
# TODO - add colour to each dot so can follow it
plot_trials = torch.tensor(torch.linspace(0, n_trials*n_epochs, 50),
                            dtype=torch.long)

for i in plot_trials[0:-1]:
    plt.scatter(results[i, :, 0], results[i, :, 1])
    # plt.scatter(results[-1, :, 0], results[-1, :, 2])
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.pause(.5)

# %% plot supervised

results = torch.stack(model.units_pos_trace, dim=0)

active_ws = torch.sum(abs(model.fc1.weight) > 0, axis=0, dtype=torch.bool)

# group
plt.scatter(results[-1, active_ws, 0], results[-1, active_ws, 1])
# plt.scatter(results[-1, active_ws, 0], results[-1, active_ws, 2])
plt.xlim([-.1, 1.1])
plt.ylim([-.1, 1.1])    
plt.show()

# over time
plot_trials = torch.tensor(torch.linspace(0, n_epochs * 8, 50),
                            dtype=torch.long)

for i in plot_trials[0:-1]:
    plt.scatter(results[i, active_ws, 0], results[i, active_ws, 1])
    # plt.scatter(results[-1, :, 0], results[-1, :å, 2])
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.pause(.5)

# attn
# plt.plot(torch.stack(model.attn_trace, dim=0))
# plt.show()

plt.plot(torch.stack(model.attn_trace[0:40], dim=0))
plt.show()

plt.plot(torch.stack(model.attn_trace, dim=0))
plt.show()

plt.plot(torch.stack(model.fc1_w_trace, dim=0)[0:20, 0, :])
plt.show()
plt.plot(torch.stack(model.fc1_w_trace, dim=0)[0:20, 1, :])
plt.show()

# # unit-based attn
# active_ws = torch.sum(abs(model.fc1.weight) > 0, axis=0, dtype=torch.bool)
# active_ws_ind = torch.nonzero(active_ws)

# for i in active_ws_ind:
#     plt.plot(torch.squeeze(torch.stack(model.attn_trace, dim=0)[:, i]))
#     plt.show()




# torch.stack(model.attn_trace[0:20], dim=0)

# torch.stack(model.winners_trace[0:20], dim=0)


# # xx1=torch.stack(model.dist_trace, dim=0)
# # yy1=torch.stack(model.act_trace, dim=0)


# plt.plot(xx-xx1)
# plt.plot(yy-yy1)

