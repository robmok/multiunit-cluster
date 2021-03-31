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
import warnings


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
        if self.attn_type[0:4] == 'dime':
            self.attn = (torch.nn.Parameter(
                torch.ones(n_dims, dtype=torch.float) * (1 / n_dims)))
            # normalize attn to 1, in case not set correctly above
            self.attn.data = (
                        self.attn.data / torch.sum(self.attn.data))
        elif self.attn_type[0:4] == 'unit':
            self.attn = (
                torch.nn.Parameter(torch.ones([n_units, n_dims],
                                              dtype=torch.float) * (1 / n_dims)))
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
        # - winning_units is like active_units before, but winning on that
        # trial, since active is define by connection weight ~=0
        # mask for winning clusters
        self.winning_units = torch.zeros(n_units, dtype=torch.bool)
        # self.mask = torch.zeros([n_classes, n_units], dtype=torch.bool)
        # # do i need this? - i think no, just to make starting weights 0
        with torch.no_grad():
            self.fc1.weight.mul_(self.mask)

    def forward(self, x):
        # compute activations of clusters here. stim x clusterpos x attn

        # distance measure. *attn works for both dimensional or unit-based
        dim_dist = abs(x - self.units_pos)
        dist = _compute_dist(dim_dist, self.attn, self.params['r'])

        # compute attention-weighted dist & activation (based on similarity)
        act = _compute_act(dist, self.params['c'], self.params['p'])

        # edited self.winning_units as index to active_ws - since normalize by all active units, not just winners.
        # pretty sure previous was wrong. but this also doesn't work as expected...
        active_ws = torch.sum(abs(self.fc1.weight) > 0, axis=0,
                              dtype=torch.bool)
        norm_units = False
        if norm_units:
            beta = self.params['beta']

            act.data[active_ws] = (
                (act.data[active_ws]**beta) /
                (torch.sum(act.data[active_ws]**beta)))

        units_output = act * self.winning_units

        # save cluster positions and activations
        # self.units_pos_trace.append(self.units_pos.detach().clone())
        # self.units_act_trace.append(units_output.detach().clone())
        self.units_act_trace.append(units_output[active_ws].detach().clone())

        # save attn weights
        if not self.attn_type[-5] == 'local':
            self.attn_trace.append(self.attn.detach().clone())

        # association weights / NN
        out = self.fc1(units_output)
        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out.detach().clone())

        # convert to response probability
        pr = self.softmax(self.params['phi'] * out)

        return out, pr


def train(model, inputs, output, n_epochs, loss_type='cross_entropy',
          shuffle=False):

    if loss_type == 'humble_teacher':
        criterion = humble_teacher
    elif loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
    if model.attn_type[-5:] != 'local':
        p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
        params = [p_fc1, p_attn]
    else:
        params = [p_fc1]

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
        # torch.manual_seed(5)
        if shuffle:
            shuffle_ind = torch.randperm(len(inputs))
            inputs_ = inputs[shuffle_ind]
            output_ = output[shuffle_ind]
        else:
            inputs_ = inputs
            output_ = output
        for x, target in zip(inputs_, output_):
            
            # TMP - testing
            # x=inputs_[np.mod(itrl-8, 8)]
            # target=output_[np.mod(itrl-8, 8)]
            # x=inputs_[itrl]
            # target=output_[itrl]
            
            # find winners
            # first: only connected units (assoc ws ~0) can be winners
            # - any weight > 0 = connected/active unit (so sum over out dim)
            active_ws = torch.sum(abs(model.fc1.weight) > 0, axis=0,
                                  dtype=torch.bool)

            if torch.sum(model.fc1.weight == 0) == 0:  # no units to recruit
                warnings.warn("No more units to recruit")

            # find units with largest activation that are connected
            dim_dist = abs(x - model.units_pos)
            dist = _compute_dist(dim_dist, model.attn, model.params['r'])
            act = _compute_act(dist, model.params['c'], model.params['p'])
            act[~active_ws] = 0  # not connected, no act

            # get top k winners
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[win_ind] == 0):
                win_ind = win_ind[act[win_ind] != 0]

            # if itrl > 0:  # checking
            #     model.dist_trace.append(dist[win_ind][0].detach().clone())
            #     model.act_trace.append(act[win_ind][0].detach().clone())

            # define winner mask
            win_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
            win_mask[:, win_ind] = True
            # this goes into forward. if ~active, no out
            model.winning_units = torch.zeros(n_units, dtype=torch.bool)
            model.winning_units[win_ind] = True

            # save acts
            # model.units_act_trace.append(act[win_ind].detach().clone())

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            # zero out gradient for masked connections
            with torch.no_grad():
                model.fc1.weight.grad.mul_(win_mask)
                if model.attn_type == 'unit':  # mask other clusters' attn
                    model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)

            # if local attn - clear attn grad computed above
            if model.attn_type[-5:] == 'local':
                model.attn.grad[:] = 0

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

                # if use local attention update - gradient ascent to unit acts
                if model.attn_type[-5:] == 'local':
                    # compute act of winners and compute gradient for attn ws
                    # spell out _compute_dist to compute gradient
                    # - note win_ind indexes winners; else non-winners added
                    # if model.params['r'] > 1:
                    #     ind = torch.sum(
                    #         abs(x - model.units_pos[win_ind]), axis=1) > 0
                    # else:
                    #     ind = range(len(win_ind))

                    # # index the winners with dist > 0 to be updated
                    # # - if 1/2 winners on stim and 1/2 not, can't selective upd
                    # ind = torch.sum(
                    #         abs(x - model.units_pos[win_ind]), axis=1) > 0
                    # # win_ind_1 = win_ind[ind]

                    # act_1 = _compute_act(
                    #     (torch.sum(model.attn *
                    #                (abs(x - model.units_pos[win_ind[ind]])
                    #                 ** model.params['r']), axis=1) **
                    #      (1/model.params['r'])),
                    #     model.params['c'], model.params['p'])


                    # NEW changing win_ind - wta winner only
                    # - only works with 
                    win_ind = win_mask[0]  # wta_mask[0] / ind[0][0] both work - wta winner
                    lose_ind = (win_mask[0] == False) & active_ws

                    # compute gradient based on activation of winners *minus*
                    # losing units.
                    act_1 = (
                        _compute_act(
                            (torch.sum(model.attn *
                                    (abs(x - model.units_pos[win_ind])
                                    ** model.params['r']), axis=1) **
                              (1/model.params['r'])), model.params['c'],
                            model.params['p']) - 

                        torch.mean(_compute_act(  # mean or sum? - mean might make more sense here, many more losers. However, losers smaller acts, so sum is needed to counteract the winners (in Cluster this is the case)
                            (torch.sum(model.attn *
                                        (abs(x - model.units_pos[lose_ind])
                                        ** model.params['r']), axis=1) **
                              (1/model.params['r'])), model.params['c'],
                            model.params['p']))
                        )

                    # compute gradient
                    for i in range(len(act_1)):
                        act_1[i].backward(retain_graph=True)
                    if len(act_1):  # if any
                        # model.attn.grad = model.attn.grad / len(act_1)  # - ah, this makes the gradients smaller. commenting it out makes it look good..
                        model.attn.data += (
                            model.params['lr_attn'] * model.attn.grad)


                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                if model.attn_type[0:4] == 'dime':
                    model.attn.data = (
                        model.attn.data / torch.sum(model.attn.data))
                elif model.attn_type[0:4] == 'unit':
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

                # save updated attn weights, unit pos
                # model.attn_trace.append(model.attn.detach().clone())
                model.units_pos_trace.append(model.units_pos.detach().clone())

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out.data) == target
            trial_ptarget[itrl] = pr[target]

            # Recruit cluster, and update model
            if (torch.tensor(recruit) and
                torch.sum(model.fc1.weight == 0) > 0):  # if no units, stop
                # 1st trial: select closest k inactive units
                if itrl == 0:  # no active weights / 1st trial
                    act = _compute_act(
                        dist, model.params['c'], model.params['p'])
                    _, recruit_ind = (
                        torch.topk(act,
                                   int(model.n_units * model.params['k'])))
                    # since topk takes top even if all 0s, remove the 0 acts
                    if torch.any(act[recruit_ind] == 0):
                        recruit_ind = recruit_ind[act[recruit_ind] != 0]
                # recruit and REPLACE k units that mispredicted
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
                # keep units that predicted correctly
                # - should work, but haven't tested since it happens rarely with currently structures
                if itrl > 0:
                    model.winning_units[win_ind[~mispred_units]] = True
                model.units_pos[recruit_ind] = x  # place at curr stim
                # model.mask[:, active_ws] = True  # new clus weights
                model.recruit_units_trl.append(itrl)

                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                with torch.no_grad():
                    win_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
                    win_mask[:, model.winning_units] = True  # update w winners
                    model.fc1.weight.grad.mul_(win_mask)
                    if model.attn_type == 'unit':
                        model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)
                if model.attn_type[-5:] == 'local':
                    model.attn.grad[:] = 0  # clear grad

                optimizer.step()

                # # if use local attention update - gradient ascent to unit acts
                # if model.attn_type[-5:] == 'local':
                #     # compute act of winners and compute gradient for attn ws
                #     # - most of the time zero since it's on the stimulus
                #     # - except when replacing. existing ones not on stim

                #     # index the winners with dist > 0 to be updated
                #     ind = torch.sum(
                #         abs(x - model.units_pos[recruit_ind]), axis=1) > 0

                #     act_1 = _compute_act(
                #         (torch.sum(model.attn *
                #                    (abs(x - model.units_pos[recruit_ind[ind]])
                #                     ** model.params['r']), axis=1) **
                #          (1/model.params['r'])),
                #         model.params['c'], model.params['p'])

                #     # compute gradient
                #     for i in range(len(act_1)):
                #         act_1[i].backward(retain_graph=True)
                #     if len(act_1):  # if any
                #         model.attn.grad = model.attn.grad / len(act_1)  # len(recruit_ind) or len(act_1) - still take the recruit_ind len?
                #         model.attn.data += (
                #             model.params['lr_attn'] * model.attn.grad)

                # # sum attention weights to 1
                # model.attn.data = torch.clamp(model.attn.data, min=0.)
                # if model.attn_type[0:4] == 'dime':
                #     model.attn.data = (
                #         model.attn.data / torch.sum(model.attn.data))
                # elif model.attn_type[0:4] == 'unit':
                #     model.attn.data = (
                #         model.attn.data /
                #         torch.sum(model.attn.data, dim=1, keepdim=True)
                #         )

                # update units - double update rule
                update = (
                    (x - model.units_pos[model.winning_units]) *
                    model.params['lr_clusters']
                    )
                model.units_pos[model.winning_units] += update

                # - step 2 - winners update towards self
                winner_mean = torch.mean(
                    model.units_pos[model.winning_units], axis=0)
                update = (
                    (winner_mean - model.units_pos[model.winning_units]) *
                    model.params['lr_clusters_group'])
                model.units_pos[model.winning_units] += update
                
                model.units_pos_trace.append(model.units_pos.detach().clone())

            # tmp
            # model.winners_trace.append(model.units_pos[model.winning_units][0])

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
            win_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
            win_mask[:, win_ind] = True

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
    # since sqrt of 0 returns nan for gradient, need this bit
    # e.g. euclid, can't **(1/2)
    if r > 1:
        d = torch.zeros(len(dim_dist))
        ind = torch.sum(dim_dist, axis=1) > 0
        dim_dist_tmp = dim_dist[ind]
        d[ind] = torch.sum(attn_w * (dim_dist_tmp ** r), axis=1)**(1/r)
    else:
        d = torch.sum(attn_w * (dim_dist**r), axis=1)**(1/r)
    return d


def _compute_act(dist, c, p):
    """ c = 1  # ALCOVE - specificity of the node - free param
        p = 2  # p=1 exp, p=2 gauss
    """
    # return torch.exp(-c * (dist**p))
    return c * torch.exp(-c * dist)  # sustain-like


# loss functions
def humble_teacher(output, target, n_classes=2):
    '''
    If multiple_tasks or output classes>2, need specify n_classes
    (unlike cross_entropy - so I had an if statement for loss=criterion...)
    '''
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
problem = 0
stim = six_problems[problem]
stim = torch.tensor(stim, dtype=torch.float)
inputs = stim[:, 0:-1]
output = stim[:, -1].long()  # integer

# # # continuous - note: need shuffle else it solves it with 1 clus
# mu1 = [-.5, .25]
# var1 = [.0185, .065]
# cov1 = -.005
# mu2 = [-.25, -.6]
# var2 = [.0125, .005]
# cov2 = .005

# model details
attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
n_units = 500
n_dims = inputs.shape[1]
# nn_sizes = [clus_layer_width, 2]  # only association weights at the end
loss_type = 'cross_entropy'
# c_recruit = 'feedback'  # feedback or loss_thresh

# top k%. so .05 = top 5%
k = .05

# SHJ
# - do I  want to save trace for both clus_pos upadtes? now just saving at the end of both updates

# trials, etc.
n_epochs = 32 # 40

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': 3,  # node specificity - 6.
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 3.5,  # response parameter, non-negative
    'lr_attn': .015,  # .005 / .05 / .001. SHJ - .01
    'lr_nn': .05,  # .1. .01 actually better, c=6. cont - .15. for fitting SHJ pattern, lr_nn=.01, 
    'lr_clusters': .015,  # .25
    'lr_clusters_group': .0,  # .95
    'k': k
    }

# new local attn
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .3,
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': .8, # (k * n_units)**-.05,  # .995**(k * n_units), #  2/np.log(k * n_units),  # norm by k units -  k * n_units
    'beta': 1.,
    'lr_attn': .01,
    'lr_nn': .2,
    'lr_clusters': .01,
    'lr_clusters_group': .05,
    'k': k
    }

# # new local attn - cluster competition
# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': .7,
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 15.5,
#     'beta': 1.,
#     'lr_attn': .01,
#     'lr_nn': .2,
#     'lr_clusters': .01,
#     'lr_clusters_group': .05,
#     'k': k
#     }

# for fitting SHJ pattern
# c=1-4 works. >6 then type I initially slower than II... 
# if p=1 seems always type I slower than II...? 

# why?
# wondering if this is because i don't normalize cluster activations, type II has more clusters
# no - others have even more and are slower! maybe coz type 1 clusters move and attn weights have to go down (unless II).. - faster attn? - yes! .005 was too slow

# ok now if keep learning, type III/V screw up like before, attn weights to 0. p doesnt matter.
# - maybe it's not the local attention rule...?
# - checked Cluster.py with wta and it works fine - with same params
# matched all here even k=1 unit...  what is different? FIND OUT

# these params works to match SHJ pattern with sustain-like activation func
# - with 8 trials per block
# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': 1,
#     'p': 1,  # p=1 exp, p =2 gauss
#     'phi': .6,
#     'lr_attn': .02, # .025, .033,
#     'lr_nn': .25, # .15,  # .25
#     'lr_clusters': .1, #.197
#     'lr_clusters_group': .0,  # .95
#     'k': k
#     }


# # trying shj with cluster competition
# params = {            
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': 1.,
#     'p': 1,  # p=1 exp, p =2 gauss
#     'beta': 1.1,
#     'phi': 6,
#     'lr_attn': .005,
#     'lr_nn': .15, # .15,  # .25
#     'lr_clusters': .1, # .01, .05
#     'lr_clusters_group': .1,
#     'k': k
#     }
        
model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=False)

# print(epoch_acc)
# print(epoch_ptarget)

# active_ws = torch.sum(abs(model.fc1.weight) > 0, axis=0, dtype=torch.bool)
# # print(np.around(model.units_pos.detach().numpy()[active_ws], decimals=2))
# print(np.unique(np.around(model.units_pos.detach().numpy()[active_ws], decimals=2), axis=0))
# # print(np.unique(np.around(model.attn.detach().numpy()[active_ws], decimals=2), axis=0))
# print(model.attn)

print(model.recruit_units_trl)
# print(len(model.recruit_units_trl))


wd = '/Users/robert.mok/Documents/Postdoc_cambridge_2020/multiunit-cluster_figs'

# plot for several k values (.01, .05, .1, .2?), several n_units (1, 1000, 10000, 1000000) - for n=1, k doesn't matter

# pr target
plt.plot(1 - epoch_ptarget.detach())
plt.ylim([0, .5])

# plt.plot(1 - trial_ptarget.detach()[0:16])
# plt.ylim([0, 1])

# if problem == 0:
#     pt = []
# pt.append(1 - epoch_ptarget.detach())

# figname = os.path.join(wd,
#                        'SHJ_prt_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
#                            problem, k, n_units, params['lr_attn'], n_epochs))
# plt.savefig(figname)
# plt.show()

# # attention weights
# plt.plot(torch.stack(model.attn_trace, dim=0))
# # figname = os.path.join(wd,
# #                        'SHJ_attn_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
# #                            problem, k, n_units, params['lr_attn'], n_epochs))
# # plt.savefig(figname)
# plt.show()

# # unit positions
# results = torch.stack(model.units_pos_trace, dim=0)[-1, active_ws]
# plt.scatter(results[:, 0], results[:, 1])
# # plt.xlim([-1, 1])
# # plt.ylim([-1, 1])
# plt.gca().set_aspect('equal', adjustable='box')
# # plt.axis('equal')

# figname = os.path.join(wd,
#                        'SHJ_unitspos2D_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
#                            problem, k, n_units, params['lr_attn'], n_epochs))
# plt.savefig(figname)
# plt.show()


# plt.plot(torch.stack(pt).T)

# %% shj cluster_wta

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


niter = 1
n_epochs = 16  # 32, 8 trials per block. 16 if 16 trials per block
pt_all = torch.zeros([niter, 6, n_epochs])

# run multiple iterations
for i in range(niter):

    # six problems

    for problem in np.array([4]):  # range(6):  # 
    
        stim = six_problems[problem]
        stim = torch.tensor(stim, dtype=torch.float)
        inputs = stim[:, 0:-1]
        output = stim[:, -1].long()  # integer
        
        # 16 per trial
        inputs = inputs.repeat(2, 1)
        output = output.repeat(2).T

        # model details
        attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
        n_units = 500
        n_dims = inputs.shape[1]
        loss_type = 'cross_entropy'
        k = .01  # top k%. so .05 = top 5%

        # these params works to match SHJ pattern with sustain-like activation func
        # k=.01
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 1,
        #     'p': 1,  # p=1 exp, p =2 gauss
        #     'phi': .8,  # n_units 500=0.6. 
        #     'lr_attn': .033, # .025, .033,
        #     'lr_nn': .15, # .15,  # .25
        #     'lr_clusters': .05, # .01, .05
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # better in cluster, better here with minor edits - higher c, higher attn
        # - k=.01, n_units=500 ok
        # - k=.01, n_units=1000 - why is this different...
        # --> when c is > 1.5, n_units makes a difference. 1.25 more similar?
        # potential issue, interaction etween lr_nn and c
        # - looks like when lr_nn >.2, c>1.5, nunits>1000, everything too fast,
        # order goes out of whack
        # - solution - lower lr_nn, higher c, phi
        # - BUT with higher k value (e.g. .05), c>1 does the above again.
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 1.6,  #  1.6 w/ k=.01, c=<1 if k=.05
        #     'p': 1,  # p=1 exp, p =2 gauss
        #     'phi': .7,  # n_units 500=0.6. 
        #     'lr_attn': .03, # .025, .033,
        #     'lr_nn': .05, # .15
        #     'lr_clusters': .01,  # .01
        #     'lr_clusters_group': .1,  # .1 - doesn't change for shj?
        #     'k': k
        #     }


        # new local attn
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .2, # .1 # n_unit=500, .5 w phi=1.5, .8 w phi=1; 1000; so keep c same works, just phi
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 1.25, # . 75. if 100 units, phi=1.25. .05 with c = 2/3. .75 with c=.5      .3.5 (k * n_units)**-.05,  # .995**(k * n_units), #  2/np.log(k * n_units),  # norm by k units -  k * n_units
            'beta': 1.,
            'lr_attn': .01*5,  # .05
            'lr_nn': .175*5,
            'lr_clusters': .05,
            'lr_clusters_group': .1,
            'k': k
            }
        # # trying with higher c - flipping 1& 6
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 1,  # high=1.75. nunits=100, low=1.75 high=3.25
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': .5,
        #     'beta': 1.,
        #     'lr_attn': .01,  # .05
        #     'lr_nn': .1,
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }



        # # new local attn + cluster comp
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .2,  # .2
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 200,  # 200
        #     'beta': .1,  # .1
        #     'lr_attn': .1,  # .01
        #     'lr_nn': .2,  # .2
        #     'lr_clusters': .01,
        #     'lr_clusters_group': .15,
        #     'k': k
        #     }
        

        # # testing - higher c, type 6 fast, type 1 slow. OK
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 3,  # 1, 2, 3
        #     'p': 1,  # p=1 exp, p =2 gauss
        #     'phi': .5,
        #     'lr_attn': .01,
        #     'lr_nn': .1,  # .1
        #     'lr_clusters': .05,
        #     'lr_clusters_group': .4,
        #     'k': k
        #     }

        # # k=.05, lr_nn needs to be lower. e.g. .05. phi lower to show effects when c>3
        # # - note, c=1 is v slow now. but c>3 is fast, and separation of 1/6/others is clear
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 3,
        #     'p': 1,  # p=1 exp, p =2 gauss
        #     'phi': .35,  #  .35 when c=3, 
        #     'lr_attn': .02, # .025, .033,
        #     'lr_nn': .035, # 
        #     'lr_clusters': .05, # .197
        #     'lr_clusters_group': .1,  # .1, .4, same
        #     'k': k
        #     }

        model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle=False)

        pt_all[i, problem] = 1 - epoch_ptarget.detach()

        print(model.recruit_units_trl)
    
plt.plot(pt_all.mean(axis=0).T)
plt.ylim([0., 0.55])
plt.gca().legend(('1','2','3','4','5','6'))
plt.show()

# the human data from nosofsky, et al. replication
shj = (
    np.array([[0.211, 0.025, 0.003, 0., 0., 0., 0., 0.,
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
               0.172, 0.128, 0.139, 0.117, 0.103, 0.098, 0.106, 0.106]])
    )

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(shj.T)
# ax[0].set_ylim([0., .55])
# ax[0].set_aspect(17)
# ax[1].plot(pt_all.mean(axis=0).T)
# ax[1].set_ylim([0., .55])
# ax[1].legend(('1', '2', '3', '4', '5', '6'), fontsize=7)
# ax[1].set_aspect(17)

# fig, ax = plt.subplots(1, 1)
# ax.plot(shj.T, 'k')
# ax.plot(pt_all.mean(axis=0).T, 'o-')
# # ax.plot(pt_all[0:10].mean(axis=0).T, 'o-')
# ax.set_ylim([0., .55])
# ax.legend(('1', '2', '3', '4', '5', '6', '1', '2', '3', '4', '5', '6'), fontsize=7)


plt.plot(torch.stack(model.attn_trace, dim=0))
plt.ylim([0.15, 0.45])
plt.show()

# %% unsupervised

# spatial / unsupervised
# looks like k is key for number of virtual clusters that come up. smaller k = more; larger k = fewer clusters 
# lr_group has to be large-ish, else virtual clusters don't form (scattered).
# lr_group has to be > lr_clusters, else virtual cluster don't form. but not too high else clusters go toward centre

# - i think the learning rates might lead to more/less grid like patterns - check which matters more (can use banino's grid code)
# - need reduction of lr over time?

# To check
# - one thing i see from plotting over time is that clusters change sometimes change across virtual clusters. need lower lr?
# looks like less later on though. maybe ok?

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


# lr = lr_clusters[5]  # >.1 [3/4/5]
# i = torch.nonzero(lr_clusters == lr)
# for j in range(len(lr_group)):
#     plt.scatter(results[:, 0, i, j], results[:, 1, i, j])
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.show()

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

