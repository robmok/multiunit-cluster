#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:56:14 2021

Multi-unit clustering model - implementing multiple banks of units to model
anterior vs posterior hippocampus.

This is done by having one banks of units (one model) with a high c parameter
and another bank of units with a low c parameter, which is a specificity param
meaning more 'specific' place fields vs more 'broad' place fields.

Idea is that SHJ problems that require attention learning will benefit from a
lower c parameter, since it needs to learn attention weights to focus on
relevant and ignore irrelevant features (e.g. Type 1), whereas others that can
immediately benefit from the cluster being on the stimulus (e.g. type 6) will
benefit from a high c param. Rule-plus-exceptions also should benefit a bit,
so they will be faster than type I.

Type II is a bit more weird, and maybe this is because it benefits from being
on the exactly stimulus, but also needs to learn attention. Often close to type
1 (slower learning), but sometimes closer to rule-plus-exception types. Maybe
because those also need some differential attention weighting.

@author: robert.mok
"""

# import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
# import itertools as it
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


def train(model, inputs, output, n_epochs, shuffle=False, lesions=None):

    criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
    if model.attn_type[-5:] != 'local':
        p_attn = {'params': [model.attn], 'lr': model.params['lr_attn']}
        params = [p_fc1, p_attn]
    else:
        params = [p_fc1]

    optimizer = optim.SGD(params, lr=model.params['lr_nn'])  # , momentum=0.)

    # save accuracy
    itrl = 0
    n_trials = len(inputs) * n_epochs
    trial_acc = torch.zeros(n_trials)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros(n_trials)
    epoch_ptarget = torch.zeros(n_epochs)

    # randomly lesion n units at ntimepoints
    if lesions:
        model.lesion_units = []  # save which units were lesioned
        if lesions['gen_rand_lesions_trials']:
            lesion_trials = (
                torch.randint(n_trials,
                              (int(n_trials * lesions['pr_lesion_trials']),)))
            model.lesion_trials = lesion_trials  # save which were lesioned
        else:
            lesion_trials = lesions['lesion_trials']

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

            # lesion trials
            if lesions:
                if torch.any(itrl == lesion_trials):
                    # find active ws, randomly turn off n units (n_lesions)
                    w_ind = np.nonzero(active_ws)
                    les = w_ind[torch.randint(w_ind.numel(),
                                              (lesions['n_lesions'],))]
                    model.lesion_units.append(les)
                    with torch.no_grad():
                        model.fc1.weight[:, les] = 0

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
            model.winning_units = torch.zeros(model.n_units, dtype=torch.bool)
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

                    # NEW changing win_ind - wta winner only
                    # - only works with wta
                    win_ind = win_mask[0]  # wta_mask[0] / ind[0][0] both work
                    lose_ind = (win_mask[0] == 0) & active_ws

                    # compute gradient based on activation of winners *minus*
                    # losing units.
                    act_1 = (
                        torch.sum(_compute_act(
                            (torch.sum(model.attn *
                                       (abs(x - model.units_pos[win_ind])
                                        ** model.params['r']), axis=1) **
                             (1/model.params['r'])), model.params['c'],
                            model.params['p'])) -

                        torch.sum(_compute_act(
                            (torch.sum(model.attn *
                                       (abs(x - model.units_pos[lose_ind])
                                        ** model.params['r']), axis=1) **
                             (1/model.params['r'])), model.params['c'],
                            model.params['p']))
                        )

                    # compute gradient
                    act_1.backward(retain_graph=True)
                    # model.attn.data += (
                    #     model.params['lr_attn'] * model.attn.grad)
                    # divide grad by n active units (scales to any n_units)
                    model.attn.data += (
                        model.params['lr_attn'] *
                        (model.attn.grad / model.n_units))

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
                model.winning_units = (
                    torch.zeros(model.n_units, dtype=torch.bool))
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

                # update units positions - double update rule
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

            itrl += 1

            if torch.sum(model.fc1.weight == 0) == 0:  # no units to recruit
                warnings.warn("No more units to recruit")

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        epoch_ptarget[epoch] = trial_ptarget[itrl-len(inputs):itrl].mean()

    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget


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

# %%



