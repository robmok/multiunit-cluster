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
import matplotlib.pyplot as plt
# import itertools as it
import warnings


class MultiUnitCluster(nn.Module):
    def __init__(self, n_units, n_dims, n_banks, attn_type, k, params=None):
        super(MultiUnitCluster, self).__init__()
        self.attn_type = attn_type
        self.n_units = n_units
        self.n_total_units = n_units * n_banks
        self.n_dims = n_dims
        self.n_banks = n_banks
        self.softmax = nn.Softmax(dim=0)
        self.active_units = torch.zeros(self.n_total_units, dtype=torch.bool)

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
        self.units_pos = torch.zeros(
            [self.n_total_units, n_dims], dtype=torch.float)

        # randomly scatter
        self.units_pos = torch.rand(
            [self.n_total_units, n_dims], dtype=torch.float)

        # attention weights - 'dimensional' = ndims / 'unit' = clusters x ndim
        if self.attn_type[0:4] == 'dime':
            self.attn = (
                torch.nn.Parameter(
                    torch.ones([n_dims, n_banks], dtype=torch.float)
                    * (1 / 3))
            )
            # normalize attn to 1, in case not set correctly above
            self.attn.data = (self.attn.data
                              / torch.sum(self.attn.data, dim=0).T)

        # network to learn association weights for classification
        n_classes = 2  # n_outputs
        self.fc1 = nn.Linear(self.n_total_units, n_classes, bias=False)
        self.fc1.weight = torch.nn.Parameter(
            torch.zeros([n_classes, self.n_total_units]))

        # mask for updating attention weights based on winning units
        # - winning_units is like active_units before, but winning on that
        # trial, since active is define by connection weight ~=0
        # mask for winning clusters
        self.winning_units = torch.zeros(self.n_total_units, dtype=torch.bool)

        # # do i need this? - i think no, just to make starting weights 0
        # with torch.no_grad():
        #     self.fc1.weight.mul_(self.winning_units)

        # masks for each bank - in order to only update one model at a time
        self.bmask = torch.zeros([n_banks, self.n_total_units],
                                 dtype=torch.bool)
        bank_ind = (
            torch.linspace(0, self.n_total_units, n_banks+1, dtype=torch.int)
            )
        # mask[ibank] will only include units from that bank.
        for ibank in range(n_banks):
            self.bmask[ibank, bank_ind[ibank]:bank_ind[ibank + 1]] = True

    def forward(self, x):

        # compute activations. stim x unit_pos x attn

        # distance measure
        dim_dist = abs(x - self.units_pos)
        dist = _compute_dist(dim_dist, self.attn, self.params['r'])

        # compute attention-weighted dist & activation (based on similarity)
        act = _compute_act(dist, self.params['c'], self.params['p'])

        # bmask - remove acts in wrong bank, sum over banks (0s for wrong bank)
        units_output = torch.sum(act * self.winning_units * self.bmask, axis=0)

        # save cluster positions and activations
        # self.units_pos_trace.append(self.units_pos.detach().clone())
        self.units_act_trace.append(
            units_output[self.active_units].detach().clone())

        # output across all banks / full model
        out = [self.fc1(units_output)]
        # get pr for the average phi value for the full model's pr
        pr = [self.softmax(np.mean(model.params['phi']) * out[0])]

        # get outputs for each nbank
        for ibank in range(self.n_banks):
            units_output_tmp = units_output.clone()
            units_output_tmp[~self.bmask[ibank]] = 0  # rmv other bank units

            # out & convert to response probability
            out.append(self.fc1(units_output_tmp))
            pr.append(self.softmax(model.params['phi'][ibank] * out[ibank+1]))

        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out)

        return out, pr


def train(model, inputs, output, n_epochs, shuffle=False, lesions=None):

    criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}

    # for local attn, just need p_fc1 with all units connected
    prms = [p_fc1]

    optimizer = optim.SGD(prms, lr=model.params['lr_nn'])  # same lr now

    # save accuracy
    itrl = 0
    n_trials = len(inputs) * n_epochs
    trial_acc = torch.zeros(n_trials)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros([model.n_banks + 1, n_trials])
    epoch_ptarget = torch.zeros([model.n_banks + 1, n_epochs])

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

            # testing
            x=inputs_[np.mod(itrl-8, 8)]
            target=output_[np.mod(itrl-8, 8)]
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
            act[:, ~model.active_units] = 0  # not connected, no act

            # bank mask
            # - extra safe: eg. at start no units, dont recruit from wrong bank
            act[~model.bmask] = -.01  # negative so never win

            # get top k winners
            _, win_ind = (
                torch.topk(act, int(model.n_units * model.params['k']),
                           dim=1)
                )

            # since topk takes top even if all 0s, remove the 0 acts
            # - indexing a bit confusing now since multiple banks, but works
            if torch.any(act[:, win_ind] == 0):
                win_ind_tmp = []
                for ibank in range(model.n_banks):
                    win_ind_tmp.append(
                        win_ind[ibank, act[ibank, win_ind[ibank]] != 0]
                        )

                # this assumes there will be the same number per bank
                # TODO - think if need allow diff numbers
                # TODO - not sure if this works other than 1st trial. if other
                # units ok, will there be 0 units? maybe this is ok?... ++
                win_ind = torch.zeros([model.n_banks, len(win_ind_tmp[0])])
                for ibank in range(model.n_banks):
                    win_ind[ibank] = win_ind_tmp[ibank]

            # define winner mask
            model.winning_units[:] = 0  # clear
            if win_ind.numel() > 0:  # only 1st trial = 0?
                model.winning_units[win_ind] = True  # goes to forward function
            win_mask = model.winning_units.repeat((len(model.fc1.weight), 1))

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            loss = criterion(out[0].unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            # zero out gradient for masked connections
            with torch.no_grad():
                model.fc1.weight.grad.mul_(win_mask)
                # if model.attn_type == 'unit':  # mask other clusters' attn
                #     model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)

            # if local attn - clear attn grad computed above
            if model.attn_type[-5:] == 'local':
                model.attn.grad[:] = 0

            # update model - if inc/recruit a cluster, don't update here
            # if incorrect, recruit
            if ((not torch.argmax(out[0].data) == target) or
               (torch.all(out[0].data == 0))):  # if incorrect
                recruit = True
            else:
                recruit = False

            # if not recruit, update model
            if recruit:
                pass
            else:
                optimizer.step()

                if model.attn_type[-5:] == 'local':

                    # wta only
                    # win_ind = model.winning_units
                    # lose_ind = (model.winning_units == 0) & model.active_units
                    for ibank in range(model.n_banks):

                        win_ind_b = (model.winning_units
                                     & model.bmask[ibank].squeeze())
                        lose_ind = (
                            (model.winning_units == 0)
                            & model.active_units
                            & model.bmask[ibank].squeeze()
                            )

                        # compute grad based on act of winners *minus* losers
                        act_1 = (
                            torch.sum(
                                _compute_act(
                                    _compute_dist(
                                        abs(x - model.units_pos[win_ind_b]),
                                        model.attn[:, ibank],
                                        model.params['r']),
                                    model.params['c'][ibank],
                                    model.params['p']))

                            - torch.sum(
                                _compute_act(
                                    _compute_dist(
                                        abs(x - model.units_pos[lose_ind]),
                                        model.attn[:, ibank],
                                        model.params['r']),
                                    model.params['c'][ibank],
                                    model.params['p']))
                            )

                        # compute gradient
                        act_1.backward(retain_graph=True)
                        # divide grad by n active units (scales to any n_units)
                        model.attn.data[:, ibank] += (
                            torch.tensor(model.params['lr_attn'][ibank])
                            * (model.attn.grad[:, ibank] / model.n_units))  # should this be n_units per bank? edited now, but was n_total_units before

                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                if model.attn_type[0:4] == 'dime':
                    model.attn.data = (
                        model.attn.data / torch.sum(model.attn.data, dim=0).T
                        )

                # save updated attn ws
                model.attn_trace.append(model.attn.detach().clone())

                # update units pos w multiple banks - double update rule
                for ibank in range(model.n_banks):
                    units_ind = model.winning_units & model.bmask[ibank]
                    update = (
                        (x - model.units_pos[units_ind])
                        * model.params['lr_clusters'][ibank]
                        )
                    model.units_pos[units_ind] += update

                    # - step 2 - winners update towards self
                    winner_mean = torch.mean(
                        model.units_pos[units_ind], axis=0)
                    update = (
                        (winner_mean - model.units_pos[units_ind])
                        * model.params['lr_clusters_group'][ibank])
                    model.units_pos[units_ind] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out[0].data) == target
            for ibank in range(model.n_banks + 1):
                trial_ptarget[ibank, itrl] = pr[ibank][target]

            # Recruit cluster, and update model
            if (torch.tensor(recruit) and
                    torch.sum(model.fc1.weight == 0) > 0):  # if no units, stop

                # 1st trial - select closest k inactive units
                if itrl == 0:
                    act = _compute_act(
                        dist, model.params['c'], model.params['p'])
                    act[~model.bmask] = -.01  # negative so never win

                    _, recruit_ind = (
                        torch.topk(act,
                                   int(model.n_units
                                       * model.params['k']), dim=1)
                        )

                    # since topk takes top even if all 0s, remove the 0 acts
                    # - shouldn't happen since 1st trial, but in case
                    if torch.any(act[:, recruit_ind] == 0):
                        r_ind_tmp = []
                        for ibank in range(model.n_banks):
                            r_ind_tmp.append(
                                recruit_ind[
                                    ibank, act[ibank, recruit_ind[ibank]] != 0]
                                )
                        # this assumes there will be the same number per bank
                        recruit_ind = torch.zeros([model.n_banks,
                                                   len(r_ind_tmp[0])])
                        for ibank in range(model.n_banks):
                            recruit_ind[ibank] = r_ind_tmp[ibank]

                # recruit and REPLACE k units that mispredicted
                else:
                    # this works only if same number in each bank
                    # TODO - think if need allow diff numbers (e.g. with list)
                    mispred_units = torch.argmax(
                        model.fc1.weight[:, win_ind].detach(), dim=0) != target

                    # select closest n_mispredicted inactive units
                    n_mispred_units = mispred_units.shape[1]  # per bank
                    act = _compute_act(
                        dist, model.params['c'], model.params['p'])
                    act[~model.bmask] = -.01  # negative so never win
                    act[:, model.active_units] = 0  # REMOVE all active units
                    # find closest units excluding the active units to recruit
                    _, recruit_ind = (
                        torch.topk(act, n_mispred_units))
                    # since topk takes top even if all 0s, remove the 0 acts
                    # - again, to be safe
                    if torch.any(act[:, recruit_ind] == 0):
                        r_ind_tmp = []
                        for ibank in range(model.n_banks):
                            r_ind_tmp.append(
                                recruit_ind[
                                    ibank, act[ibank, recruit_ind[ibank]] != 0]
                                )
                        # this assumes there will be the same number per bank
                        recruit_ind = torch.zeros([model.n_banks,
                                                   len(r_ind_tmp[0])])
                        for ibank in range(model.n_banks):
                            recruit_ind[ibank] = r_ind_tmp[ibank]

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
                loss = criterion(out[0].unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                with torch.no_grad():
                    win_mask[:] = 0  # clear
                    win_mask[:, model.winning_units] = True  # update w winners
                    model.fc1.weight.grad.mul_(win_mask)
                    # if model.attn_type == 'unit':
                    #     model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)
                if model.attn_type[-5:] == 'local':
                    model.attn.grad[:] = 0  # clear grad

                optimizer.step()

                # TODO - local attn update? omitted for now

                # save updated attn ws - save even if not update
                model.attn_trace.append(model.attn.detach().clone())

                # update units pos w multiple banks - double update rule
                for ibank in range(model.n_banks):
                    units_ind = model.winning_units & model.bmask[ibank]
                    update = (
                        (x - model.units_pos[units_ind])
                        * model.params['lr_clusters'][ibank]
                        )
                    model.units_pos[units_ind] += update

                    # - step 2 - winners update towards self
                    winner_mean = torch.mean(
                        model.units_pos[units_ind], axis=0)
                    update = (
                        (winner_mean - model.units_pos[units_ind])
                        * model.params['lr_clusters_group'][ibank])
                    model.units_pos[units_ind] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())

            itrl += 1

            if torch.sum(model.fc1.weight == 0) == 0:  # no units to recruit
                warnings.warn("No more units to recruit")

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        for ibank in range(model.n_banks + 1):
            epoch_ptarget[ibank, epoch] = (
                trial_ptarget[ibank, itrl-len(inputs):itrl].mean()
                )

    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget


def _compute_dist(dim_dist, attn_w, r):
    # since sqrt of 0 returns nan for gradient, need this bit
    # e.g. euclid, can't **(1/2)
    if r > 1:
        # d = torch.zeros(len(dim_dist))
        # ind = torch.sum(dim_dist, axis=1) > 0
        # dim_dist_tmp = dim_dist[ind]
        # d[ind] = torch.sum(attn_w * (dim_dist_tmp ** r), axis=1)**(1/r)
        pass
    else:
        # compute distances weighted by 2 banks of attn weights
        if len(attn_w.shape) > 1:  # if more than 1 bank
            # - all dists computed but for each bank, only n_units shd be used
            d = torch.zeros([len(dim_dist), model.n_banks])
            for ibank in range(model.n_banks):
                d[:, ibank] = (
                    torch.sum(attn_w[:, ibank] * (dim_dist**r), axis=1)
                    ** (1/r)
                    )
        else:
            d = torch.sum(attn_w * (dim_dist**r), axis=1) ** (1/r)
    return d


def _compute_act(dist, c, p):
    """
    - dist is n_banks x n_total_units, and params['c'] size is n_banks, so
    can just multiple by banks
    - sustain activation function
    """
    if torch.tensor(c).numel() > 1:
        act = torch.transpose(
            torch.tensor(c) * torch.exp(-torch.tensor(c) * dist), 0, 1)
    else:
        act = c * torch.exp(-c * dist)
    return act

# %%


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

# set problem
problem = 0
stim = six_problems[problem]
stim = torch.tensor(stim, dtype=torch.float)
inputs = stim[:, 0:-1]
output = stim[:, -1].long()  # integer


# model details
attn_type = 'dimensional_local'  # dimensional, unit, dimensional_local
n_units = 500
# n_dims = inputs.shape[1]
n_dims = 3
loss_type = 'cross_entropy'

# top k%. so .05 = top 5%
k = .05

# n banks of units
n_banks = 2

# SHJ

# trials, etc.
n_epochs = 16

# new local attn - scaling lr
lr_scale = (n_units * k) / 1

# merged - some kept same across banks
params = {
    'r': 1,  # 1=city-block, 2=euclid
    'c': [1.5, 3.5],  # flips works with this even w same phi! prev [.8, 3.5]
    'p': 1,  # p=1 exp, p=2 gauss
    'phi': [1.5, 1.5],  # can flip work if phi is same - so 2 banks are competiting at the outputs. yes
    'beta': 1,
    'lr_attn': [.25, .002],  # this scales at grad computation now
    'lr_nn': .025/lr_scale,  # scale by n_units*k - keep the same for now
    'lr_clusters': [.01, .01],
    'lr_clusters_group': [.1, .1],
    'k': k
    }

# testing type 1 vs 6
# - flips for type 1
params = {
    'r': 1,
    'c': [1.5, 2.5],  # [1.5, 2.5] / [1.5, 3.5]
    'p': 1,
    'phi': [1.5, 1.5],
    'beta': 1,
    'lr_attn': [.25, .02],  # [.25, .02]
    'lr_nn': .025/lr_scale,
    'lr_clusters': [.01, .01],
    'lr_clusters_group': [.1, .1],
    'k': k
    }

model = MultiUnitCluster(n_units, n_dims, n_banks, attn_type, k, params=params)

model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget = train(
    model, inputs, output, n_epochs, shuffle=False)

# pr target
plt.plot(1 - epoch_ptarget.T.detach())
plt.ylim([0, .5])
plt.show()

# attention weights
fig, ax = plt.subplots(1, 2)
ax[0].plot(torch.stack(model.attn_trace, dim=0)[:, :, 0])
ax[0].set_ylim([torch.stack(model.attn_trace, dim=0).min()-.01,
                torch.stack(model.attn_trace, dim=0).max()+.01])
ax[1].plot(torch.stack(model.attn_trace, dim=0)[:, :, 1])
ax[1].set_ylim([torch.stack(model.attn_trace, dim=0).min()-.01,
                torch.stack(model.attn_trace, dim=0).max()+.01])
plt.show()
