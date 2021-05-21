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
        self.active_units = torch.zeros(n_units, dtype=torch.bool)
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

        # attention weights - 'dimensional' = ndims / 'unit' = clusters x ndim
        if self.attn_type[0:4] == 'dime':
            self.attn = (torch.nn.Parameter(
                torch.ones(n_dims, dtype=torch.float) * (1 / n_dims)))
            # normalize attn to 1, in case not set correctly above
            self.attn.data = (
                        self.attn.data / torch.sum(self.attn.data))
        elif self.attn_type[0:4] == 'unit':
            self.attn = (
                torch.nn.Parameter(
                    torch.ones([n_units, n_dims], dtype=torch.float)
                    * (1 / n_dims)))
            # normalize attn to 1, in case not set correctly above
            self.attn.data = (
                self.attn.data /
                torch.sum(self.attn.data, dim=1, keepdim=True))

        # network to learn association weights for classification
        n_classes = 2  # n_outputs
        self.fc1 = nn.Linear(n_units, n_classes, bias=False)
        self.fc1.weight = torch.nn.Parameter(torch.zeros([n_classes, n_units]))

        # mask for NN
        # - probably don't need this
        # - below I used win_mask to define this, then used win_mask to create
        # model.winning_units... actually probably can just have one.
        # self.mask = torch.zeros([n_classes, n_units], dtype=torch.bool)

        # mask for updating attention weights based on winning units
        # - winning_units is like active_units before, but winning on that
        # trial, since active is define by connection weight ~=0
        # mask for winning clusters
        self.winning_units = torch.zeros(n_units, dtype=torch.bool)

        # # do i need this? - i think no, just to make starting weights 0
        # with torch.no_grad():
        #     self.fc1.weight.mul_(self.winning_units)

    def forward(self, x):

        # compute activations. stim x unit_pos x attn

        # distance measure. *attn works for both dimensional or unit-based
        dim_dist = abs(x - self.units_pos)
        dist = _compute_dist(dim_dist, self.attn, self.params['r'])

        # compute attention-weighted dist & activation (based on similarity)
        act = _compute_act(dist, self.params['c'], self.params['p'])

        norm_units = False
        if norm_units:
            beta = self.params['beta']
            act.data[self.active_units] = (
                (act.data[self.active_units]**beta) /
                (torch.sum(act.data[self.active_units]**beta)))

        units_output = act * self.winning_units

        # save cluster positions and activations
        # self.units_pos_trace.append(self.units_pos.detach().clone())
        self.units_act_trace.append(
            units_output[self.active_units].detach().clone())

        # association weights / NN
        out = self.fc1(units_output)
        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out.detach().clone())

        # convert to response probability
        pr = self.softmax(self.params['phi'] * out)

        return out, pr


def train(model, inputs, output, n_epochs, shuffle=False, shuffle_seed=None,
          lesions=None):

    criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
    if model.attn_type[-5:] != 'local':
        p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
        prms = [p_fc1, p_attn]
    else:
        prms = [p_fc1]

    optimizer = optim.SGD(prms, lr=model.params['lr_nn'])  # , momentum=0.)

    # save accuracy
    itrl = 0
    n_trials = len(inputs) * n_epochs
    trial_acc = torch.zeros(n_trials)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros(n_trials)
    epoch_ptarget = torch.zeros(n_epochs)

    # lesion units during learning
    if lesions:
        model.lesion_units = []  # save which units were lesioned
        if lesions['gen_rand_lesions_trials']:  # lesion at randomly timepoints
            lesion_trials = (
                torch.randint(n_trials,
                              (int(n_trials * lesions['pr_lesion_trials']),)))
            model.lesion_trials = lesion_trials  # save which were lesioned
        else:  # lesion at pre-specified timepoints
            lesion_trials = lesions['lesion_trials']

    model.train()

    if shuffle_seed:
        torch.manual_seed(shuffle_seed)

    for epoch in range(n_epochs):
        if shuffle:
            shuffle_ind = torch.randperm(len(inputs))
            inputs_ = inputs[shuffle_ind]
            output_ = output[shuffle_ind]
        else:
            inputs_ = inputs
            output_ = output
        for x, target in zip(inputs_, output_):

            # testing
            # x=inputs_[np.mod(itrl-8, 8)]
            # target=output_[np.mod(itrl-8, 8)]
            # x=inputs_[itrl]
            # target=output_[itrl]

            # lesion trials
            if lesions:
                if torch.any(itrl == lesion_trials):
                    # find active ws, randomly turn off n units (n_lesions)
                    w_ind = np.nonzero(model.active_units)
                    les = w_ind[torch.randint(w_ind.numel(),
                                              (lesions['n_lesions'],))]
                    model.lesion_units.append(les)
                    with torch.no_grad():
                        model.fc1.weight[:, les] = 0

            # find winners:largest acts that are connected (model.active_units)
            dim_dist = abs(x - model.units_pos)
            dist = _compute_dist(dim_dist, model.attn, model.params['r'])
            act = _compute_act(dist, model.params['c'], model.params['p'])
            act[~model.active_units] = 0  # not connected, no act

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
            model.winning_units[:] = 0  # clear
            model.winning_units[win_ind] = True  # goes to forward function
            win_mask = model.winning_units.repeat((len(model.fc1.weight), 1))

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

                    # NEW changing win_ind - wta winner only
                    # - only works with wta
                    win_ind = model.winning_units
                    lose_ind = (model.winning_units == 0) & model.active_units

                    # compute gradient based on activation of winners *minus*
                    # losing units.
                    # act_1 = (
                    #     torch.sum(_compute_act(
                    #         (torch.sum(model.attn
                    #                     * (abs(x - model.units_pos[win_ind])
                    #                       ** model.params['r']), axis=1)
                    #           ** (1/model.params['r'])), model.params['c'],
                    #         model.params['p']))

                    #     - torch.sum(_compute_act(
                    #         (torch.sum(model.attn
                    #                     * (abs(x - model.units_pos[lose_ind])
                    #                       ** model.params['r']), axis=1)
                    #           ** (1/model.params['r'])), model.params['c'],
                    #         model.params['p']))
                    #     )

                    act_1 = (
                        torch.sum(
                            _compute_act(
                                _compute_dist(
                                    abs(x - model.units_pos[win_ind]),
                                    model.attn, model.params['r']),
                                model.params['c'], model.params['p']))

                        - torch.sum(
                            _compute_act(
                                _compute_dist(
                                    abs(x - model.units_pos[lose_ind]),
                                    model.attn, model.params['r']),
                                model.params['c'], model.params['p']))
                        )

                    # compute gradient
                    act_1.backward(retain_graph=True)
                    # divide grad by n active units (scales to any n_units)
                    model.attn.data += (
                        model.params['lr_attn']
                        * (model.attn.grad / model.n_units))

                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                if model.attn_type[0:4] == 'dime':
                    model.attn.data = (
                        model.attn.data / torch.sum(model.attn.data)
                        )
                elif model.attn_type[0:4] == 'unit':
                    model.attn.data = (
                        model.attn.data
                        / torch.sum(model.attn.data, dim=1, keepdim=True)
                        )

                # save updated attn ws
                model.attn_trace.append(model.attn.detach().clone())

                # update units - double update rule
                # - step 1 - winners update towards input
                update = (
                    (x - model.units_pos[win_ind])
                    * model.params['lr_clusters']
                    )
                model.units_pos[win_ind] += update

                # - step 2 - winners update towards self
                winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
                update = (
                    (winner_mean - model.units_pos[win_ind])
                    * model.params['lr_clusters_group']
                    )
                model.units_pos[win_ind] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out.data) == target
            trial_ptarget[itrl] = pr[target]

            # Recruit cluster, and update model
            if (torch.tensor(recruit) and
                torch.sum(model.fc1.weight == 0) > 0):  # if no units, stop

                # 1st trial - select closest k inactive units
                if itrl == 0:
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
                    act[model.active_units] = 0  # REMOVE all active units
                    # find closest units excluding the active units to recruit
                    _, recruit_ind = (
                        torch.topk(act, n_mispred_units))
                    # since topk takes top even if all 0s, remove the 0 acts
                    if torch.any(act[recruit_ind] == 0):
                        recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit n_mispredicted units
                model.active_units[recruit_ind] = True  # set ws to active
                model.winning_units[:] = 0  # clear
                model.winning_units[recruit_ind] = True
                # keep units that predicted correctly
                # - should work, but haven't tested since it happens rarely
                # with currently structures
                if itrl > 0:
                    model.winning_units[win_ind[~mispred_units]] = True
                model.units_pos[recruit_ind] = x  # place at curr stim
                model.recruit_units_trl.append(itrl)

                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                with torch.no_grad():
                    win_mask[:] = 0  # clear
                    win_mask[:, model.winning_units] = True  # update w winners
                    model.fc1.weight.grad.mul_(win_mask)
                    if model.attn_type == 'unit':
                        model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)
                if model.attn_type[-5:] == 'local':
                    model.attn.grad[:] = 0  # clear grad

                optimizer.step()

                # for recruited units, gradient is zero
                # however, if replacing units, old units will have a grad
                # TODO
                # - problem looks like the losers have high act now (so act_1
                # can be negative, screws things up sometimes....)

                # if model.attn_type[-5:] == 'local':
                #     win_ind = win_mask[0],
                #     lose_ind = (win_mask[0] == 0) & model.active_units

                #     # gradient based on activation of winners minus losers
                #     act_1 = (
                #         torch.sum(_compute_act(
                #             (torch.sum(model.attn *
                #                         (abs(x - model.units_pos[win_ind])
                #                         ** model.params['r']), axis=1) **
                #               (1/model.params['r'])), model.params['c'],
                #             model.params['p'])) -

                #         torch.sum(_compute_act(
                #             (torch.sum(model.attn *
                #                         (abs(x - model.units_pos[lose_ind])
                #                         ** model.params['r']), axis=1) **
                #               (1/model.params['r'])), model.params['c'],
                #             model.params['p']))
                #         )

                #     # compute gradient
                #     act_1.backward(retain_graph=True)
                #     # divide grad by n active units (scales to any n_units)
                #     model.attn.data += (
                #         model.params['lr_attn'] *
                #         (model.attn.grad / model.n_units))

                # save updated attn ws
                # model.attn_trace.append(model.attn.detach().clone())

                # update units positions - double update rule
                update = (
                    (x - model.units_pos[model.winning_units])
                    * model.params['lr_clusters']
                    )
                model.units_pos[model.winning_units] += update

                # - step 2 - winners update towards self
                winner_mean = torch.mean(
                    model.units_pos[model.winning_units], axis=0)
                update = (
                    (winner_mean - model.units_pos[model.winning_units])
                    * model.params['lr_clusters_group'])
                model.units_pos[model.winning_units] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())

            # tmp
            # model.winners_trace.append(model.units_pos[model.winning_units][0])

            itrl += 1

            if torch.sum(model.fc1.weight == 0) == 0:  # no units to recruit
                warnings.warn("No more units to recruit")

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
            # find winners with largest activation - all connected
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
            # win_mask = torch.zeros(model.mask.shape, dtype=torch.bool)
            # win_mask[:, win_ind] = True
            # above is old code, below is new. test. actually.. need this?
            model.winning_units[:] = 0  # clear
            model.winning_units[win_ind] = True  # goes to forward function
            win_mask = model.winning_units.repeat((len(model.fc1.weight), 1))

            # learn
            # update units - double update rule
            # - step 1 - winners update towards input
            update = (
                (x - model.units_pos[win_ind]) * model.params['lr_clusters'])
            model.units_pos[win_ind] += update

            # - step 2 - winners update towards self
            winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
            update = (
                (winner_mean - model.units_pos[win_ind])
                * model.params['lr_clusters_group'])
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
        d = torch.sum(attn_w * (dim_dist**r), axis=1) ** (1/r)
    return d


def _compute_act(dist, c, p):
    """ c = 1  # ALCOVE - specificity of the node - free param
        p = 2  # p=1 exp, p=2 gauss
    """
    # return torch.exp(-c * (dist**p))
    return c * torch.exp(-c * dist)  # sustain-like


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

# 16 per trial
inputs = inputs.repeat(2, 1)
output = output.repeat(2).T
        
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
n_epochs = 16

# params = {
#     'r': 1,  # 1=city-block, 2=euclid
#     'c': 3,  # node specificity - 6.
#     'p': 1,  # p=1 exp, p=2 gauss
#     'phi': 3.5,  # response parameter, non-negative
#     'lr_attn': .015,  # .005 / .05 / .001. SHJ - .01
#     'lr_nn': .05,  # .1. .01 actually better, c=6. cont - .15. for fitting SHJ pattern, lr_nn=.01, 
#     'lr_clusters': .015,  # .25
#     'lr_clusters_group': .0,  # .95
#     'k': k
#     }

# new local attn - scaling lr
lr_scale = (n_units * k) / 1

params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': .9, # w/ attn grad normalized, c can be large now
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 18.5,
    'beta': 1.,
    'lr_attn': .15, # this scales at grad computation now
    'lr_nn': .01/lr_scale,  # scale by n_units*k
    'lr_clusters': .01,
    'lr_clusters_group': .1,
    'k': k
    }

# shj params
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': 1.,  # w/ attn grad normalized, c can be large now
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': 12.5,
    'beta': 1.,
    'lr_attn': .15,  # this scales at grad computation now
    'lr_nn': .015/lr_scale,  # scale by n_units*k
    'lr_clusters': .01,
    'lr_clusters_group': .1,
    'k': k
    }

# lesioning
lesions = None  # if no lesions
# lesions = {
#     'n_lesions': 10,  # n_lesions per event
#     'gen_rand_lesions_trials': False,  # generate lesion events at random times
#     'pr_lesion_trials': .01,  # if True, set this
#     'lesion_trials': torch.tensor([20])  # if False, set lesion trials
#     }

model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=False, lesions=lesions)

# # print(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2))
# print(np.unique(np.around(model.units_pos.detach().numpy()[model.active_units], decimals=2), axis=0))
# # print(np.unique(np.around(model.attn.detach().numpy()[model.active_units], decimals=2), axis=0))
# print(model.attn)

print(model.recruit_units_trl)
# print(len(model.recruit_units_trl))

# wd='/Users/robert.mok/Documents/Postdoc_cambridge_2020/multiunit-cluster_figs'
# plot for several k values (.01, .05, .1, .2?)
# several n_units (1, 1000, 10000, 1000000) - for n=1, k doesn't matter

# pr target
plt.plot(1 - epoch_ptarget.detach())
plt.ylim([0, .5])
plt.show()

# # attention weights
plt.plot(torch.stack(model.attn_trace, dim=0))
# figname = os.path.join(wd,
#                        'SHJ_attn_{}_k{}_nunits{}_lra{}_epochs{}.png'.format(
#                            problem, k, n_units, params['lr_attn'], n_epochs))
# plt.savefig(figname)
plt.show()

# # unit positions
# results = torch.stack(model.units_pos_trace, dim=0)[-1, model.active_units]
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


# explore lesion units ++ 
# model.units_pos[model.lesion_units[0]] # inspect which units were lesions on lesion trial 0

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

    for problem in range(6):  # [0, 5]: #  np.array([4]):

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
        k = .05  # top k%. so .05 = top 5%

        # scale lrs - params determined by n_units=100, k=.01. n_units*k=1
        lr_scale = (n_units * k) / 1

        # new local attn
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': .8,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 10.5,
            'beta': 1.,
            'lr_attn': .15,  # this scales at grad computation now
            'lr_nn': .025/lr_scale,  # scale by n_units*k
            'lr_clusters': .01,
            'lr_clusters_group': .1,
            'k': k
            }
    
        # trying with higher c - flipping 1& 6
        # - works well - needs lr_attn to be v slow, then type 6>1 (flipped)
        # now type II also can be slow, types 3-5 faster - as brad predicted
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': 3.5,  # low = 1; med = 2.2; high = 3.5+
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 1.5, 
        #     'beta': 1.,
        #     'lr_attn': .002,  # if too slow, type 1 recruits 4 clus..
        #     'lr_nn': .025/lr_scale,  # scale by n_units*k
        #     'lr_clusters': .01,
        #     'lr_clusters_group': .1,
        #     'k': k
        #     }

        # # new local attn + cluster comp
        # params = {
        #     'r': 1,  # 1=city-block, 2=euclid
        #     'c': .01,  # .2
        #     'p': 1,  # p=1 exp, p=2 gauss
        #     'phi': 100,  # 200
        #     'beta': 1.5,  # .1
        #     'lr_attn': .4/lr_scale,  
        #     'lr_nn': .3,
        #     'lr_clusters': .01,
        #     'lr_clusters_group': .15,
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

fig, ax = plt.subplots(2, 1)
ax[0].plot(shj.T)
ax[0].set_ylim([0., .55])
ax[0].set_aspect(17)
ax[1].plot(pt_all.mean(axis=0).T)
ax[1].set_ylim([0., .55])
ax[1].legend(('1', '2', '3', '4', '5', '6'), fontsize=7)
ax[1].set_aspect(17)
plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.plot(shj.T, 'k')
# ax.plot(pt_all.mean(axis=0).T, 'o-')
# # ax.plot(pt_all[0:10].mean(axis=0).T, 'o-')
# ax.set_ylim([0., .55])
# ax.legend(('1', '2', '3', '4', '5', '6', '1', '2', '3', '4', '5', '6'), fontsize=7)

# plt.plot(torch.stack(model.attn_trace, dim=0))
# plt.ylim([0.15, 0.45])
# plt.show()

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

# group
plt.scatter(results[-1, model.active_units, 0], results[-1, model.active_units, 1])
# plt.scatter(results[-1, model.active_units, 0], results[-1, model.active_units, 2])
plt.xlim([-.1, 1.1])
plt.ylim([-.1, 1.1])    
plt.show()

# over time
plot_trials = torch.tensor(torch.linspace(0, n_epochs * 8, 50),
                            dtype=torch.long)

for i in plot_trials[0:-1]:
    plt.scatter(results[i, model.active_units, 0], results[i, model.active_units, 1])
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
# model.active_units = torch.sum(abs(model.fc1.weight) > 0, axis=0, dtype=torch.bool)
# active_ws_ind = torch.nonzero(model.active_units)

# for i in active_ws_ind:
#     plt.plot(torch.squeeze(torch.stack(model.attn_trace, dim=0)[:, i]))
#     plt.show()




# torch.stack(model.attn_trace[0:20], dim=0)

# torch.stack(model.winners_trace[0:20], dim=0)


# # xx1=torch.stack(model.dist_trace, dim=0)
# # yy1=torch.stack(model.act_trace, dim=0)


# plt.plot(xx-xx1)
# plt.plot(yy-yy1)


# %% lesioning experiments

problem = 4
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
k = .05  # top k%

n_epochs = 16

lesions = {
    'n_lesions': 10,  # n_lesions per event
    'gen_rand_lesions_trials': False,  # generate lesion events at random times
    'pr_lesion_trials': .01,  # if True, set this
    'lesion_trials': torch.tensor([20])  # if False, set lesion trials
    }

shuffle_seed = 5

# model = MultiUnitCluster(n_units, n_dims, attn_type, k, params=params)

# model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
#     model, inputs, output, n_epochs, shuffle=True, shuffle_seed=shuffle_seed,
#     lesions=lesions)

# print(model.recruit_units_trl)
# print(len(model.recruit_units_trl))

# # pr target
# plt.plot(1 - epoch_ptarget.detach())
# plt.ylim([0, .5])
# plt.show()

# # attention weights
# plt.plot(torch.stack(model.attn_trace, dim=0))
# plt.show()


# for All: need 1 simulation with lesions vs no lesions - w same shuffled seq
# - feed in a random number for seed: shuffle_seed = torch.randperm(n_sims)
# or torch.randperm(n_sims*5)[:n_sims] to get more nums so diff over other sims
# - HMMM you might also want the same shuffle over a set of sims, if randomly
# lesioning units or random time points!


# expt 1: n_lesions [single lesionevent] - number of units. and timing of event
# - manipulation n_lesions
# - manpulate k value and n_total units. will be affects by k most, but of course n_total interacts
# - fix / manipulate: shuffle - seed the same num across a set of sims, then
# seed another num for another set; run nset sims. this is to test same shuffle
# different lesions (since they are random which units get lesioned)
# - fix: lesion_trials at 1 time point (across a few sims, different time pt) 
# save for each sim: model.recruit_units_trl, len(model.recruit_units_trl),
# epoch_ptarget.detach(), model.attn_trace

# expt 2: n lesion events, and timing
# - first, do nlesions early, middle, late. then also do random.
# e.g. [0:10 early, 0 mid, 0 late], then [0 early, 0:10 mid, 0 late], etc.

n_sims = 10
shuffle_seeds = torch.randperm(n_sims*5)[:n_sims]

# shuffle_seeds[isim]

# things to manipulate
n_units = [20, 100, 500]
k = [.05]
n_lesions = [0, 10, 20]
lesion_trials = np.array([[40]])  # , [40], [60]]  # 1 per lesion, but do at diff times

sim_ps = []
pt = []
recruit_trial = []
attn_trace = []

# can add loop for problem in range(6)

for sim_prms in it.product(n_units, k, n_lesions, lesion_trials):
    for isim in range(n_sims):

        sim_ps.append(sim_prms)

        # shj params
        params = {
            'r': 1,  # 1=city-block, 2=euclid
            'c': 1.,  # w/ attn grad normalized, c can be large now
            'p': 1,  # p=1 exp, p=2 gauss
            'phi': 12.5,
            'beta': 1.,
            'lr_attn': .15,  # this scales at grad computation now
            'lr_nn': .015/(sim_prms[0] * sim_prms[1]),  # scale by n_units*k
            'lr_clusters': .01,
            'lr_clusters_group': .1,
            'k': sim_prms[1],
            }

        model = MultiUnitCluster(sim_prms[0], n_dims, attn_type, sim_prms[1],
                                 params=params)

        lesions = {
            'n_lesions': sim_prms[2],  # n_lesions per event
            'gen_rand_lesions_trials': False,  # lesion events at random times
            'pr_lesion_trials': .01,  # if True, set this
            'lesion_trials': torch.tensor(sim_prms[3])  # if False, set this
            }

        model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
            model, inputs, output, n_epochs, shuffle=True,
            shuffle_seed=shuffle_seeds[isim], lesions=lesions)

        pt.append(1 - epoch_ptarget.detach())
        recruit_trial.append(model.recruit_units_trl)
        attn_trace.append(torch.stack(model.attn_trace, dim=0))

# %%

# plt.plot(torch.stack(pt[0:3]).T)  # .mean(axis=0).T)
# plt.ylim([0., 0.55])
# plt.gca().legend(('0 lesions','10 lesions','20 lesions'))
# plt.show()


# with n_sims
# - it goes backwards from how sim_prms are ordered. so 

pts = torch.stack(pt)

ind_sims = []
for i in range(len(pt) // n_sims):
    ind_sims.append(torch.arange(i * n_sims, (i + 1) * n_sims))

# average over sims and plot
pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(0, 3)]  # specify sims
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('0 lesions', '10 lesions', '20 lesions'))
plt.title('20 units')
plt.show()

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(3, 6)]
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('0 lesions', '10 lesions', '20 lesions'))
plt.title('100 units')
plt.show()

pt_plot = [pts[ind_sims[i]].mean(axis=0) for i in range(6, 9)]
plt.plot(torch.stack(pt_plot).T)
plt.ylim([0., 0.55])
plt.gca().legend(('0 lesions', '10 lesions', '20 lesions'))
plt.title('500 units')
plt.show()



# attn
# TODO - why different number of trials
# plt.plot(torch.stack(attn_trace[0:3]))  # .mean(axis=0).T)
# plt.ylim([0., 0.55])
# plt.show()

# next: organise how to index
# - including: average over n_sims




# plt.gca().legend(('0 lesions','n lesions','n lesions'))
# plt.show()



# For plotting, make df?

# import pandas as pd
# df_sum = pd.DataFrame(columns=['acc', 'k', 'n_uni'ts, 'n_lesions', 'lesion trials', 'sim_num'])

