#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:12:04 2021

Multi-unit clustering  model

@author: robert.mok
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from scipy.stats import norm


class MultiUnitCluster(nn.Module):
    def __init__(self, n_units, n_dims, attn_type, k, params=None,
                 fit_params=False, start_params=False):
        super(MultiUnitCluster, self).__init__()
        self.attn_type = attn_type
        self.n_units = n_units
        self.n_dims = n_dims
        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.active_units = torch.zeros(n_units, dtype=torch.bool)

        # history
        self.attn_trace = []
        self.units_pos_trace = []
        self.units_pos_bothupd_trace = []
        self.units_act_trace = []
        self.recruit_units_trl = []
        self.fc1_w_trace = []
        self.fc1_act_trace = []

        # free params
        if params:
            self.params = params
        else:  # some defaults - not using now
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

        # if fitting params (gridsearch, mle)
        if fit_params:
            self.params = {
                'r': 1,
                'c': start_params[0],
                'p': 1,
                'phi': start_params[1],
                'lr_attn': start_params[2],
                'lr_nn': start_params[3],
                'lr_clusters': start_params[4],
                'lr_clusters_group': start_params[5],
                'k': k,
                }

        # randomly scatter units
        self.units_pos = torch.rand([n_units, n_dims], dtype=torch.float)

        # attention weights - 'dimensional' = ndims / 'unit' = clusters x ndim
        self.attn = (torch.nn.Parameter(
                torch.ones(n_dims, dtype=torch.float) * (1 / n_dims)))
        # normalize attn to 1, in case not set correctly above
        self.attn.data = (
                    self.attn.data / torch.sum(self.attn.data))

        # network to learn association weights for classification
        n_classes = 2  # n_outputs
        self.fc1 = nn.Linear(n_units, n_classes, bias=False)
        self.fc1.weight = torch.nn.Parameter(torch.zeros([n_classes, n_units]))

        # mask for updating attention weights based on winning units
        # - winning_units is like active_units before, but winning on that
        # trial, since active is define by connection weight ~=0
        # mask for winning clusters
        self.winning_units = torch.zeros(n_units, dtype=torch.bool)

    # compute forward pass / activations. stim x unit_pos x attn
    def forward(self, x):

        # distance measure. *attn works for both dimensional or unit-based
        dim_dist = abs(x - self.units_pos)
        dist = self._compute_dist(
            dim_dist, self.attn, self.params['r'])

        # compute attention-weighted dist & activation (based on similarity)
        act = self._compute_act(
            dist, self.params['c'], self.params['p'])

        units_output = act * self.winning_units

        # save cluster positions and activations
        self.units_act_trace.append(
            units_output[self.active_units].detach().clone())

        # association weights / NN
        # - include phi param into output
        # - note pytorch takes this out and computes CE loss by combining
        # nn.LogSoftmax() and nn.NLLLoss(), so logsoftmax is applied, no need
        # to apply to out here
        out = self.params['phi'] * self.fc1(units_output)

        # convert to response probability
        pr = self.softmax(self.params['phi'] * self.fc1(units_output))

        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(out.detach().clone())

        return out, pr

    def _compute_dist(self, dim_dist, attn_w, r):

        if r > 1:
            d = torch.zeros(len(dim_dist))
            ind = (torch.sum(dim_dist, axis=1) > 0)
            dim_dist_tmp = dim_dist[ind]
            d[ind] = torch.sum(attn_w * (dim_dist_tmp ** r), axis=1)**(1/r)
        else:
            d = torch.sum(attn_w * (dim_dist**r), axis=1) ** (1/r)
        return d

    def _compute_act(self, dist, c, p):
        return c * torch.exp(-c * dist)  # sustain-like

    def _norm_diff(self, a, b):  # , n_active_units):
        return a-b  # no normalization


def train(model, inputs, output, n_epochs, shuffle_seed=None, lesions=None,
          noise=None, shj_order=False):

    criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}
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
    les_units = []  # initialize
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
        shuffle_ind = torch.randperm(len(inputs))
        inputs_ = inputs[shuffle_ind]
        output_ = output[shuffle_ind]

        # 1st block, show 8 unique stim, then 8 again. after, shuffle 16
        if shj_order:
            if epoch == 0:
                shuffle_ind = torch.cat(
                    [torch.randperm(len(inputs)//2),
                      torch.randperm(len(inputs)//2) + len(inputs)//2])
                inputs_ = inputs[shuffle_ind]
                output_ = output[shuffle_ind]
            else:
                shuffle_ind = torch.randperm(len(inputs))
                inputs_ = inputs[shuffle_ind]
                output_ = output[shuffle_ind]

        for x, target in zip(inputs_, output_):

            # # testing
            # x=inputs_[np.mod(itrl-8, 8)]
            # target=output_[np.mod(itrl-8, 8)]
            # x=inputs_[itrl]
            # target=output_[itrl]

            # lesion trials
            if lesions:
                if torch.any(itrl == lesion_trials):
                    # find active ws, randomly turn off n units (n_lesions)
                    w_ind = np.nonzero(model.active_units)

                    # if lesions > active units, sel fewer
                    if lesions['n_lesions'] > len(w_ind):
                        # index all active units
                        les = w_ind
                        # if nles > active units, also lesion inactive units
                        n_more_lesions = lesions['n_lesions'] - len(w_ind)
                        # inactive units
                        w_ind_inact = np.nonzero(~model.active_units)
                        # if nmlesions > inactive units, set to ninactive units
                        # if inactive units > nmlesions, set to nmlesions
                        n_lesions_inact = min([len(w_ind_inact),
                                               n_more_lesions])

                        les_inact = np.random.choice(w_ind_inact.flatten(),
                                                     n_lesions_inact,
                                                     replace=False)

                        les = torch.vstack([les,
                                            torch.tensor([les_inact, ]).T])
                    else:
                        les = np.random.choice(w_ind.flatten(),
                                               lesions['n_lesions'],
                                               replace=False)
                        les = torch.tensor(les)

                    # note: making les units 'active' so won't re-recruit them
                    model.active_units[les] = False
                    with torch.no_grad():
                        model.fc1.weight[:, les] = 0

                    model.lesion_units.append(les.flatten())
                    # note to check which not to recruit
                    les_units = torch.hstack(model.lesion_units)

            # find winners:largest acts that are connected (model.active_units)
            dim_dist = abs(x - model.units_pos)
            dist = model._compute_dist(dim_dist, model.attn, model.params['r'])
            act = model._compute_act(
                dist, model.params['c'], model.params['p'])
            act[~model.active_units] = 0  # not connected, no act

            # get top k winners
            _, win_ind = torch.topk(act,
                                    int(model.n_units
                                        * model.params['k']))
            win_ind = win_ind
            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[win_ind] == 0):
                win_ind = win_ind[act[win_ind] != 0]

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
                win_mask = win_mask
                # print(win_mask.is_cuda)
                model.fc1.weight.grad.mul_(win_mask)
                # if model.attn_type == 'unit':  # mask other clusters' attn
                #     model.attn.grad.mul_(win_mask[0].unsqueeze(0).T)

            # if local attn - clear attn grad computed above
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

                    win_ind = model.winning_units
                    lose_ind = (model.winning_units == 0) & model.active_units

                    act_1 = model._norm_diff(
                        torch.sum(
                            model._compute_act(
                                model._compute_dist(
                                    abs(x - model.units_pos[win_ind]),
                                    model.attn, model.params['r']
                                    ),
                                model.params['c'], model.params['p']
                                )),
                        torch.sum(
                            model._compute_act(
                                model._compute_dist(
                                    abs(x - model.units_pos[lose_ind]),
                                    model.attn, model.params['r']
                                    ),
                                model.params['c'], model.params['p']
                                ))
                        )

                    # compute gradient
                    act_1.backward(retain_graph=True)
                    # divide grad by n active units (scales to any n_units)
                    model.attn.data += (
                        model.params['lr_attn']
                        # * model.attn.grad)  # scale above
                        * (model.attn.grad / model.active_units.sum()))  # scale here but edit norm_diff to just be a-b

                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                model.attn.data = (
                    model.attn.data / torch.sum(model.attn.data)
                    )

                # save updated attn ws
                model.attn_trace.append(model.attn.detach().clone())

                # update units - double update rule
                # - step 1 - winners update towards input
                update = (
                    (x - model.units_pos[win_ind])
                    * model.params['lr_clusters']
                    )

                # add noise to updates
                if noise:
                    update += (
                        torch.tensor(
                            norm.rvs(loc=noise['update1'][0],
                                     scale=noise['update1'][1],
                                     size=(len(update), model.n_dims)))
                        * model.params['lr_clusters']
                            )

                model.units_pos[win_ind] += update

                # store unit positions after both upds
                model.units_pos_bothupd_trace.append(
                    model.units_pos.detach().clone())

                # - step 2 - winners update towards self
                winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
                update = (
                    (winner_mean - model.units_pos[win_ind])
                    * model.params['lr_clusters_group']
                    )

                # add noise to 2nd update?
                if noise:
                    update += (
                        torch.tensor(
                            norm.rvs(loc=noise['update2'][0],
                                     scale=noise['update2'][1],
                                     size=(len(update), model.n_dims)))
                        * model.params['lr_clusters_group']
                            )

                model.units_pos[win_ind] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())
                model.units_pos_bothupd_trace.append(
                    model.units_pos.detach().clone())  # store both upds

            # save acc per trial
            trial_acc[itrl] = torch.argmax(out.data) == target
            trial_ptarget[itrl] = pr[target]

            # Recruit cluster, and update model
            if (torch.tensor(recruit)
                    and torch.sum(model.fc1.weight == 0) > 0
                    and len(les_units) < model.n_units):  # if no more units, stop

                # 1st trial - select closest k inactive units
                if itrl == 0:
                    act = model._compute_act(
                        dist, model.params['c'], model.params['p'])
                    _, recruit_ind = (
                        torch.topk(act,
                                   int(model.n_units
                                       * model.params['k'])))
                    recruit_ind = recruit_ind

                    # since topk takes top even if all 0s, remove the 0 acts
                    if torch.any(act[recruit_ind] == 0):
                        recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit and REPLACE k units that mispredicted
                else:
                    if len(win_ind):  # else all lesioned
                        mispred_units = (
                            torch.argmax(model.fc1.weight[:, win_ind].detach(),
                                         dim=0) != target
                            )
                    else:
                        mispred_units = torch.tensor(0)

                    # select closest n_mispredicted inactive units
                    n_mispred_units = mispred_units.sum()
                    act = model._compute_act(
                        dist, model.params['c'], model.params['p'])
                    act[model.active_units] = 0  # REMOVE all active units
                    act[les_units] = 0  # REMOVE all lesioned units
                    # find closest units excluding the active units to recruit
                    _, recruit_ind = (
                        torch.topk(act, n_mispred_units))
                    recruit_ind = recruit_ind

                    # since topk takes top even if all 0s, remove the 0 acts
                    if torch.any(act[recruit_ind] == 0):
                        recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit n_mispredicted units
                model.active_units[recruit_ind] = True  # set ws to active
                model.winning_units[:] = 0  # clear
                model.winning_units[recruit_ind] = True
                # keep units that predicted correctly
                if itrl > 0 and len(win_ind):  # else all lesioned
                    model.winning_units[win_ind[~mispred_units]] = True
                model.units_pos[recruit_ind] = x  # place at curr stim
                model.recruit_units_trl.append(itrl)

                # add noise to recruited positions
                if noise:
                    model.units_pos[recruit_ind] += (
                        torch.tensor(
                            norm.rvs(loc=noise['recruit'][0],
                                     scale=noise['recruit'][1],
                                     size=(len(recruit_ind), model.n_dims)))
                            )

                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                loss = criterion(out.unsqueeze(0), target.unsqueeze(0))
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

                # update attn - new 2022 put back in
                win_ind = model.winning_units
                lose_ind = (model.winning_units == 0) & model.active_units

                act_1 = model._norm_diff(
                    torch.sum(
                        model._compute_act(
                            model._compute_dist(
                                abs(x - model.units_pos[win_ind]),
                                model.attn, model.params['r']
                                ),
                            model.params['c'], model.params['p']
                            )),
                    torch.sum(
                        model._compute_act(
                            model._compute_dist(
                                abs(x - model.units_pos[lose_ind]),
                                model.attn, model.params['r']
                                ),
                            model.params['c'], model.params['p']
                            ))
                    )

                # compute gradient
                act_1.backward(retain_graph=True)
                # divide grad by n active units (scales to any n_units)
                model.attn.data += (
                    model.params['lr_attn']
                    # * model.attn.grad)  # if scale above
                    * (model.attn.grad / model.active_units.sum()))

                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                model.attn.data = (
                    model.attn.data / torch.sum(model.attn.data)
                    )
                # save updated attn ws - even if don't update
                model.attn_trace.append(model.attn.detach().clone())

                # update units positions - double update rule
                update = (
                    (x - model.units_pos[model.winning_units])
                    * model.params['lr_clusters']
                    )

                # add noise to updates
                if noise:
                    update += (
                        torch.tensor(
                            norm.rvs(loc=noise['update1'][0],
                                     scale=noise['update1'][1],
                                     size=(len(update), model.n_dims)))
                        * model.params['lr_clusters']
                            )

                model.units_pos[model.winning_units] += update

                # store unit positions after both upds
                model.units_pos_bothupd_trace.append(
                    model.units_pos.detach().clone())

                # - step 2 - winners update towards self
                winner_mean = torch.mean(
                    model.units_pos[model.winning_units], axis=0)
                update = (
                    (winner_mean - model.units_pos[model.winning_units])
                    * model.params['lr_clusters_group'])

                # add noise to 2nd update?
                if noise:
                    update += (
                        torch.tensor(
                            norm.rvs(loc=noise['update2'][0],
                                     scale=noise['update2'][1],
                                     size=(len(update), model.n_dims)))
                        * model.params['lr_clusters_group']
                            )

                model.units_pos[model.winning_units] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())
                model.units_pos_bothupd_trace.append(
                    model.units_pos.detach().clone())

                # add noise to 2nd update?
                if noise:
                    model.units_pos[model.winning_units] += (
                        torch.tensor(
                            norm.rvs(loc=noise['update2'][0],
                                     scale=noise['update2'][1],
                                     size=(len(update), model.n_dims)))
                            )

            itrl += 1

            if torch.sum(model.fc1.weight == 0) == 0:  # no units to recruit
                warnings.warn("No more units to recruit")

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        epoch_ptarget[epoch] = trial_ptarget[itrl-len(inputs):itrl].mean()

    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget


def train_unsupervised(model, inputs, n_epochs, batch_upd=None, noise=None):
    """
    batch_upd : if not None, this is the batch number. input trials and batch
    number each time you call train_unsupervised, it will update the mean
    update for all trials

    """
    # if batch,  len(inputs) is n_trials_batch
    # make an array for updating at the end - n_units x n_trials_batch
    # - add update for each trial, then np.nansum(upd), then tensor it
    # - torch.nansum() should be coming...
    upd_pos = torch.zeros(
        model.n_units, len(inputs), inputs.shape[1]) * float('nan')
    upd_attn = torch.zeros(model.n_dims, len(inputs))

    if batch_upd is not None:
        itrl = batch_upd * len(inputs)  # assuming same ntrials/batch
        itrl_b = 0  # for upd arrays
    else:
        itrl = 0

    for epoch in range(n_epochs):
        for x in inputs:

            # testing
            # x = inputs[itrl]

            # find winners:largest acts that are connected (model.active_units)
            dim_dist = abs(x - model.units_pos)
            dist = model._compute_dist(dim_dist, model.attn, model.params['r'])
            act = model._compute_act(
                dist, model.params['c'], model.params['p'])
            act[~model.active_units] = 0  # not connected, no act

            # get top k winners
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[win_ind] == 0):
                win_ind = win_ind[act[win_ind] != 0]

            # define winner mask
            model.winning_units[:] = 0  # clear
            model.winning_units[win_ind] = True
            # win_mask = model.winning_units.repeat((len(model.fc1.weight), 1))

            # recruit
            # - if error is high (inverse of activations of winners), recruit
            # - scale to k winners. eg. min is .1 of sum of k units max act
            thresh = (
                .7 * (model.n_units * model.params['k'] * model.params['c'])
                )

            if act[win_ind].sum() < thresh:
                recruit = True
            else:
                recruit = False

            # if not recruit, update model
            if recruit:
                pass
            else:

                # update attention
                if model.params['lr_attn'] > 0:
                    win_ind = model.winning_units
                    lose_ind = (model.winning_units == 0) & model.active_units

                    act_1 = model._norm_diff(
                        torch.sum(
                            model._compute_act(
                                model._compute_dist(
                                    abs(x - model.units_pos[win_ind]),
                                    model.attn, model.params['r']
                                    ),
                                model.params['c'], model.params['p']
                                )),
                        torch.sum(
                            model._compute_act(
                                model._compute_dist(
                                    abs(x - model.units_pos[lose_ind]),
                                    model.attn, model.params['r']
                                    ),
                                model.params['c'], model.params['p']
                                ))
                        )

                    # compute gradient
                    act_1.backward(retain_graph=True)

                    if batch_upd is None:
                        # divide grad by n active units (scales to any n_units)
                        model.attn.data += (
                            model.params['lr_attn']  # [itrl]  # when annealing
                            # * model.attn.grad)  # scale above
                            * (model.attn.grad / model.active_units.sum()))

                        # ensure attention are non-negative
                        model.attn.data = torch.clamp(model.attn.data, min=0.)
                        # sum attention weights to 1
                        model.attn.data = (
                            model.attn.data / torch.sum(model.attn.data)
                            )
                        # save updated attn ws
                        model.attn_trace.append(model.attn.detach().clone())

                    else:  # batch - save the update
                        # - save the update then mean then normalise?
                        # - or get the upd, normalize attn by when it would be,
                        # and subtract that - save that as the update
                        # ... not sure if it's the same - try both

                        # save gradient
                        upd_attn[:, itrl_b] = (
                            model.params['lr_attn']  # [itrl]  # when annealing
                            * (model.attn.grad / model.n_units)
                            )

                        # # OR add grad, norm then subtract to get the norm'd upd
                        # attn_tmp = model.attn.data + (model.params['lr_attn'] # [itrl]  # when annealing
                        #                               * (model.attn.grad /
                        #                                  model.n_units))
                        # attn_tmp = torch.clamp(attn_tmp, min=0.)
                        # attn_tmp = attn_tmp / torch.sum(attn_tmp)
                        # upd_attn[:, itrl_b] = attn_tmp - model.attn.data

                # update units - double update rule
                # - step 1 - winners update towards input
                update = (
                    (x - model.units_pos[win_ind])
                    * model.params['lr_clusters'][itrl]
                    )

                # add noise to updates
                if noise:
                    update += (
                        torch.tensor(
                            norm.rvs(loc=noise['update1'][0],
                                     scale=noise['update1'][1],
                                     size=(len(update), model.n_dims)))
                        * model.params['lr_clusters'][itrl]
                            )  # added lr - same as adding noise to upd above

                if batch_upd is None:
                    model.units_pos[win_ind] += update
                else:  # save update
                    upd_pos[win_ind, itrl_b] = update

                # - step 2 - winners update towards self
                winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
                update = (
                    (winner_mean - model.units_pos[win_ind])
                    * model.params['lr_clusters_group'][itrl]
                    )

                # add noise to 2nd update?
                if noise:
                    update += (
                        torch.tensor(
                            norm.rvs(loc=noise['update2'][0],
                                     scale=noise['update2'][1],
                                     size=(len(update), model.n_dims)))
                        * model.params['lr_clusters_group'][itrl]
                        )

                if batch_upd is None:
                    model.units_pos[win_ind] += update
                    # save updated unit positions
                    model.units_pos_trace.append(
                        model.units_pos.detach().clone())
                else:  # add to the update
                    upd_pos[win_ind, itrl_b] += update

            # Recruit cluster, and update model
            # - batch - can recruit, and show activations above, just no pos
            # or attn updates.
            if (recruit and torch.sum(model.active_units == 0) > 0):

                # select closest k inactive units
                act = model._compute_act(
                    dist, model.params['c'], model.params['p'])
                act[model.active_units] = 0  # REMOVE all active units
                # find closest units excluding the active units to recruit
                _, recruit_ind = (
                    torch.topk(act, int(model.n_units * model.params['k'])))
                # since topk takes top even if all 0s, remove the 0 acts
                if torch.any(act[recruit_ind] == 0):
                    recruit_ind = recruit_ind[act[recruit_ind] != 0]

                # recruit n_units
                model.active_units[recruit_ind] = True  # set ws to active
                model.winning_units[:] = 0  # clear
                model.winning_units[recruit_ind] = True
                model.units_pos[recruit_ind] = x  # place at curr stim
                model.recruit_units_trl.append(itrl)

                # add noise to recruited positions
                if noise:
                    model.units_pos[recruit_ind] += (
                        torch.tensor(
                            norm.rvs(loc=noise['recruit'][0],
                                     scale=noise['recruit'][1],
                                     size=(len(recruit_ind), model.n_dims)))
                            )

                # save updated attn ws - even if don't update
                # if model.params['lr_attn'] > 0:
                #     model.attn_trace.append(model.attn.detach().clone())

                # # update units positions - double update rule
                # # - only matters when there is noise
                # update = (
                #     (x - model.units_pos[recruit_ind])
                #     * model.params['lr_clusters'][itrl]
                #     )

                # if batch_upd is None:
                #     model.units_pos[recruit_ind] += update
                # else:  # save update
                #     upd_pos[recruit_ind, itrl_b] = update

                # # - step 2 - winners update towards self
                # winner_mean = torch.mean(
                #     model.units_pos[recruit_ind], axis=0)
                # update = (
                #     (winner_mean - model.units_pos[recruit_ind])
                #     * model.params['lr_clusters_group'])
                # if batch_upd is None:
                #     model.units_pos[recruit_ind] += update
                #     # save updated unit positions
                #     model.units_pos_trace.append(
                #         model.units_pos.detach().clone())
                # else:  # add to the update
                #     upd_pos[recruit_ind, itrl_b] += update

                # recruit and update WITHIN batch
                update = (
                    (x - model.units_pos[recruit_ind])
                    * model.params['lr_clusters'][itrl]
                    )
                model.units_pos[recruit_ind] += update

                # - step 2 - winners update towards self
                winner_mean = torch.mean(
                    model.units_pos[recruit_ind], axis=0)
                update = (
                    (winner_mean - model.units_pos[recruit_ind])
                    * model.params['lr_clusters_group'][itrl])
                model.units_pos[recruit_ind] += update

                # save updated unit positions - if update within batch
                # model.units_pos_trace.append(
                #     model.units_pos.detach().clone())

                # # update and clear saved updates up till recruit
                # upd_pos_mean = np.nanmean(upd_pos, axis=1)
                # upd_pos_mean[np.isnan(upd_pos_mean)] = 0  # turns nans to 0
                # model.units_pos += upd_pos_mean
                # # save updated unit positions
                # # model.units_pos_trace.append(model.units_pos.detach().clone())

                # # clear
                # upd_pos[:] = float('nan')

            # save acts - may want to have it separately for recruit and upd?
            model.fc1_act_trace.append(
                act[model.winning_units].detach().clone())

            itrl += 1

            if batch_upd is not None:
                itrl_b += 1

        if torch.sum(model.active_units == 0) == 0:  # no units to recruit
            warnings.warn("No more units to recruit")

    if batch_upd is not None:
        upd_pos_mean = np.nanmean(upd_pos, axis=1)
        upd_pos_mean[np.isnan(upd_pos_mean)] = 0  # turns nans to 0
        model.units_pos += upd_pos_mean
        # save updated unit positions
        model.units_pos_trace.append(model.units_pos.detach().clone())

        if model.params['lr_attn'] > 0:
            model.attn.data += upd_attn.mean(axis=1)
            # normalise
            model.attn.data = torch.clamp(model.attn.data, min=0.)
            model.attn.data = (
                model.attn.data / torch.sum(model.attn.data)
                )
            model.attn_trace.append(model.attn.detach().clone())

    # return upd_pos, upd_attn  # if batch. actually don't need it, just upd.


def train_unsupervised_k(model, inputs, n_epochs, batch_upd=None):
    """ No recruitment, just upd closest units. Demonstrates double upd better

    batch_upd : if not None, this is the batch number. input trials and batch
    number each time you call train_unsupervised, it will update the mean
    update for all trials

    """

    upd_pos = torch.zeros(
        model.n_units, len(inputs), inputs.shape[1]) * float('nan')

    if batch_upd is not None:
        itrl = batch_upd * len(inputs)  # assuming same ntrials/batch
        itrl_b = 0  # for upd arrays
    else:
        itrl = 0

    for epoch in range(n_epochs):
        for x in inputs:

            # find winners with largest activation - all connected
            dim_dist = abs(x - model.units_pos)
            dist = model._compute_dist(dim_dist, model.attn, model.params['r'])
            act = model._compute_act(
                dist, model.params['c'], model.params['p'])

            # get top k winners
            _, win_ind = torch.topk(act,
                                    int(model.n_units * model.params['k']))
            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[win_ind] == 0):
                win_ind = win_ind[act[win_ind] != 0]

            # define winner mask
            model.winning_units[:] = 0  # clear
            model.winning_units[win_ind] = True

            # update units - double update rule
            # - step 1 - winners update towards input
            update = (
                (x - model.units_pos[win_ind])
                * model.params['lr_clusters'][itrl])

            if batch_upd is None:
                model.units_pos[win_ind] += update
            else:  # save update
                upd_pos[win_ind, itrl_b] = update

            # - step 2 - winners update towards self
            winner_mean = torch.mean(model.units_pos[win_ind], axis=0)
            update = (
                (winner_mean - model.units_pos[win_ind])
                * model.params['lr_clusters_group'])

            if batch_upd is None:
                model.units_pos[win_ind] += update
                # save updated unit positions
                model.units_pos_trace.append(
                    model.units_pos.detach().clone())
            else:  # add to the update
                upd_pos[win_ind, itrl_b] += update

            # store positions over time
            model.fc1_act_trace.append(
                act[model.winning_units].detach().clone())

            itrl += 1

            if batch_upd is not None:
                itrl_b += 1

    if batch_upd is not None:
        upd_pos_mean = np.nanmean(upd_pos, axis=1)
        upd_pos_mean[np.isnan(upd_pos_mean)] = 0  # turns nans to 0
        model.units_pos += upd_pos_mean
        # save updated unit positions
        model.units_pos_trace.append(model.units_pos.detach().clone())
