#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:12:07 2021

@author: robert.mok
"""

import torch
import torch.nn as nn
import torch.optim as optim
# import warnings


class MultiUnitClusterNBanks(nn.Module):
    def __init__(self, n_units, n_dims, n_banks, attn_type, k, params=None,
                 fit_params=False, start_params=False):
        super(MultiUnitClusterNBanks, self).__init__()
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
        self.recruit_units_trl = [[] for i in range(self.n_banks)]
        self.fc1_w_trace = []
        self.fc1_act_trace = []

        lr_scale = (self.n_units * k) / 1

        # free params
        if params:
            self.params = params
        else:
            self.params = {
                'r': 1,
                'c': [.75, 2.5],
                'p': 1,
                'phi': [1.3, 1.2],
                'beta': 1,
                'lr_attn': [.2, .002],
                'lr_nn': [.05/lr_scale, .01/lr_scale],
                'lr_clusters': [.05, .05],
                'lr_clusters_group': [.1, .1],
                'k': k
                }

        if fit_params:
            self.params = {
                'r': 1,
                'c': [start_params[0], start_params[6]],
                'p': 1,
                'phi': [start_params[1], start_params[7]],
                'beta': 1,
                'lr_attn': [start_params[2], start_params[8]],
                'lr_nn': [start_params[3]/lr_scale, start_params[9]/lr_scale],
                'lr_clusters': [start_params[4], start_params[10]],
                'lr_clusters_group': [start_params[5], start_params[11]],
                'k': k
                }

        # units
        # randomly scatter
        self.units_pos = torch.rand(
            [self.n_total_units, n_dims], dtype=torch.float)

        # attention weights - 'dimensional' = ndims / 'unit' = clusters x ndim
        self.attn = torch.nn.Parameter(torch.ones([n_dims, n_banks],
                                                  dtype=torch.float)
                                       * (1 / 3))
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

        # distance measure
        dim_dist = abs(x - self.units_pos)
        dist = self._compute_dist(dim_dist, self.attn, self.params['r'],
                                  self.n_banks)

        # compute attention-weighted dist & activation (based on similarity)
        act = self._compute_act(dist, self.params['c'], self.params['p'])

        # bmask - remove acts in wrong bank, sum over banks (0s for wrong bank)
        units_output = torch.sum(act * self.winning_units * self.bmask, axis=0)

        # save cluster positions and activations
        self.units_act_trace.append(
            units_output[self.active_units].detach().clone())

        # output
        # output across all banks / full model
        out_b = []
        pr_b = []

        # get outputs for each nbank
        for ibank in range(self.n_banks):
            units_output_tmp = units_output.clone()
            units_output_tmp[~self.bmask[ibank]] = 0  # rmv other bank units

            # out & convert to response probability
            # - note pytorch takes this out and computes CE loss by combining
            # nn.LogSoftmax() and nn.NLLLoss(), so logsoftmax is applied, no
            # need to apply to out here
            out_b.append(self.params['phi'][ibank]
                         * self.fc1(units_output_tmp))
            # save for plotting - same as exp(out) since out is logsoftmaxed
            pr_b.append(self.softmax(self.params['phi'][ibank] * out_b[ibank]))

        # add full model output to start
        out = [sum(out_b)]
        out.extend(out_b)

        pr = [self.softmax(sum(out_b))]
        pr.extend(pr_b)

        self.fc1_w_trace.append(self.fc1.weight.detach().clone())
        self.fc1_act_trace.append(torch.stack(pr).detach().clone())

        return out, pr

    def _compute_dist(self, dim_dist, attn_w, r, n_banks):
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
                d = torch.zeros([len(dim_dist), n_banks])
                for ibank in range(n_banks):
                    d[:, ibank] = (
                        torch.sum(attn_w[:, ibank] * (dim_dist**r), axis=1)
                        ** (1/r)
                        )
            else:
                d = torch.sum(attn_w * (dim_dist**r), axis=1) ** (1/r)
        return d  # **2  # squared dist

    def _compute_act(self, dist, c, p):
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


def train(model, inputs, output, n_epochs, shuffle_seed=None, lesions=None,
          noise=None, shj_order=False):   # to add noise

    criterion = nn.CrossEntropyLoss()

    # buid up model params
    p_fc1 = {'params': model.fc1.parameters()}

    # for local attn, just need p_fc1 with all units connected
    prms = [p_fc1]

    # diff lr per bank - multiply by fc1.weight.grad by lr's below
    optimizer = optim.SGD(prms, lr=1.)

    # save accuracy
    itrl = 0
    n_trials = len(inputs) * n_epochs
    trial_acc = torch.zeros(n_trials)
    epoch_acc = torch.zeros(n_epochs)
    trial_ptarget = torch.zeros([model.n_banks + 1, n_trials])
    epoch_ptarget = torch.zeros([model.n_banks + 1, n_epochs])

    # # lesion units during learning
    # if lesions:
    #     model.lesion_units = []  # save which units were lesioned
    #     if lesions['gen_rand_lesions_trials']:  # lesion at randomly timepoints
    #         lesion_trials = (
    #             torch.randint(n_trials,
    #                           (int(n_trials * lesions['pr_lesion_trials']),)))
    #         model.lesion_trials = lesion_trials  # save which were lesioned
    #     else:  # lesion at pre-specified timepoints
    #         lesion_trials = lesions['lesion_trials']

    model.train()

    if shuffle_seed:
        torch.manual_seed(shuffle_seed)

    for epoch in range(n_epochs):

        shuffle_ind = torch.randperm(len(inputs))
        inputs_ = inputs[shuffle_ind]
        output_ = output[shuffle_ind]

        # 1st block, show 8 unique stim, then 8 again. after, shuffle 16
        if shj_order:
            shuffle_ind = torch.cat(
                [torch.randperm(len(inputs)//2),
                 torch.randperm(len(inputs)//2) + len(inputs)//2])
            inputs_ = inputs[shuffle_ind]
            output_ = output[shuffle_ind]

        else:
            inputs_ = inputs
            output_ = output

        for x, target in zip(inputs_, output_):

            # # testing
            # x=inputs_[np.mod(itrl-8, 8)]
            # target=output_[np.mod(itrl-8, 8)]
            # x=inputs_[itrl]
            # target=output_[itrl]

            # # lesion trials
            # if lesions:
            #     if torch.any(itrl == lesion_trials):
            #         # find active ws, randomly turn off n units (n_lesions)
            #         w_ind = np.nonzero(model.active_units)
            #         les = w_ind[torch.randint(w_ind.numel(),
            #                                   (lesions['n_lesions'],))]
            #         model.lesion_units.append(les)
            #         with torch.no_grad():
            #             model.fc1.weight[:, les] = 0

            # find winners:largest acts that are connected (model.active_units)
            dim_dist = abs(x - model.units_pos)
            dist = model._compute_dist(dim_dist, model.attn, model.params['r'],
                                       model.n_banks)
            act = model._compute_act(
                dist, model.params['c'], model.params['p'])
            act[:, ~model.active_units] = 0  # not connected, no act

            # bank mask
            # - extra safe: eg. at start no units, dont recruit from wrong bank
            # - also useful for "torch.any(act[:, recruit_ind_flat] == 0)"
            # since there checking for 0 acts. these are not real 0s.
            act[~model.bmask] = -.01  # negative so never win

            # get top k winners
            _, win_ind = (
                torch.topk(act, int(model.n_units * model.params['k']),
                           dim=1)
                )

            # since topk takes top even if all 0s, remove the 0 acts
            if torch.any(act[:, win_ind] == 0):
                win_ind_tmp = []
                for ibank in range(model.n_banks):
                    win_ind_tmp.append(
                        win_ind[ibank, act[ibank, win_ind[ibank]] != 0]
                        )
                # win_ind as list so can have diff n per bank
                win_ind = win_ind_tmp
            else:
                win_ind = win_ind.tolist()

            # flatten
            win_ind_flat = [item for sublist in win_ind for item in sublist]

            # define winner mask
            model.winning_units[:] = 0  # clear
            if len(win_ind_flat) > 0:  # only 1st trial = 0?
                model.winning_units[win_ind_flat] = True  # goes to forward
            win_mask = model.winning_units.repeat((len(model.fc1.weight), 1))

            # learn
            optimizer.zero_grad()
            out, pr = model.forward(x)
            loss = criterion(out[0].unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            # zero out gradient for masked connections
            with torch.no_grad():
                model.fc1.weight.grad.mul_(win_mask)
                # diff learning rates per bank (set lr_nn to 1 above)
                for ibank in range(model.n_banks):
                    model.fc1.weight.grad[:, model.bmask[ibank]] = (
                        model.fc1.weight.grad[:, model.bmask[ibank]]
                        * model.params['lr_nn'][ibank]
                        )

            # if local attn - clear attn grad computed above
            model.attn.grad[:] = 0

            # update model - if inc/recruit a cluster, don't update here

            # recruit per bank
            recruit = [(torch.argmax(out[ibank + 1]) != target) or
                       (torch.all(out[ibank + 1].data == 0))
                       for ibank in range(model.n_banks)]

            rec_banks = torch.nonzero(torch.tensor(recruit))
            upd_banks = torch.nonzero(~torch.tensor(recruit))

            # if not recruit, update model
            if all(recruit):  # if no banks correct, all recruit (no upd)
                pass
            else:  # if at least one bank correct (~recruit), update

                # remove nn updates for recruiting bank
                for ibank in rec_banks:
                    model.fc1.weight.grad[:, model.bmask[ibank].squeeze()] = 0

                # update nn weights
                optimizer.step()

                # if local attn - k-wta only
                for ibank in upd_banks:

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
                            model._compute_act(
                                model._compute_dist(
                                    abs(x - model.units_pos[win_ind_b]),
                                    model.attn[:, ibank].squeeze(),
                                    model.params['r'], model.n_banks),
                                model.params['c'][ibank],
                                model.params['p']))

                        - torch.sum(
                            model._compute_act(
                                model._compute_dist(
                                    abs(x - model.units_pos[lose_ind]),
                                    model.attn[:, ibank].squeeze(),
                                    model.params['r'], model.n_banks),
                                model.params['c'][ibank],
                                model.params['p']))
                        )

                    # compute gradient
                    act_1.backward(retain_graph=True)
                    # divide grad by n active units (scales to any n_units)
                    model.attn.data[:, ibank] += (
                        torch.tensor(model.params['lr_attn'][ibank])
                        * (model.attn.grad[:, ibank]
                            / model.active_units[
                                model.bmask[ibank].squeeze()].sum()))

                # ensure attention are non-negative
                model.attn.data = torch.clamp(model.attn.data, min=0.)
                # sum attention weights to 1
                model.attn.data = (
                    model.attn.data / torch.sum(model.attn.data, dim=0).T
                    )

                # save updated attn ws
                model.attn_trace.append(model.attn.detach().clone())

                # update units pos w multiple banks - double update rule
                for ibank in upd_banks:
                    units_ind = (model.winning_units
                                 & model.bmask[ibank].squeeze())
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

            # Recruit cluster, and update model (if no avail units, stop)
            if any(recruit) and torch.sum(model.fc1.weight == 0) > 0:

                # 1st trial - select closest k inactive units for both banks
                if itrl == 0:
                    act = model._compute_act(
                        dist, model.params['c'], model.params['p'])
                    act[~model.bmask] = -.01  # negative so never win

                    _, recruit_ind = (
                        torch.topk(act,
                                   int(model.n_units
                                       * model.params['k']), dim=1)
                        )

                    recruit_ind = recruit_ind.tolist()

                # recruit and REPLACE k units that mispredicted
                else:
                    mispred_units = [
                        torch.argmax(
                            model.fc1.weight[:, win_ind[ibank]].detach(),
                            dim=0) != target for ibank in rec_banks
                        ]

                    # select closest n_mispredicted inactive units
                    # - range(len()) now since may only have 1 bank. even if >
                    # 2 banks, should work since in order of mispred_units
                    n_mispred_units = [mispred_units[ibank].sum()
                                       for ibank in range(len(rec_banks))]

                    # above nice since both will have len of 2 if 2 banks, n
                    # len of 1 is 1 bank. but below i index them with [ibank]
                    # which doesn't work
                    # - deal with the len of the list somehow... could use
                    # enumerate (count, ibanks) in the loop, so use
                    # n_mispred_units[count] / mispred_units[count].

                    act = model._compute_act(
                        dist, model.params['c'], model.params['p'])
                    act[~model.bmask] = -.01  # negative so never win
                    act[:, model.active_units] = 0  # REMOVE all active units
                    # find closest units excluding the active units to recruit
                    recruit_ind_tmp = []
                    for i, ibank in enumerate(rec_banks):
                        _, recruit_ind = (
                            torch.topk(act[ibank], n_mispred_units[i])
                            )
                        # extend opposed to append like in prev script. prev
                        # always rec n_banks, so did n_mispred_units[ibanks]
                        # but here i, so flattening works differently
                        recruit_ind_tmp.extend(recruit_ind)

                    recruit_ind = recruit_ind_tmp  # list

                # flatten
                recruit_ind_flat = [item for sublist in recruit_ind
                                    for item in sublist]

                # since topk takes top even if all 0s, remove the 0 acts
                # - note this works because i made irrelevant bank units -0.01
                # check other script for notes
                if torch.any(act[:, recruit_ind_flat] == 0):
                    r_ind_tmp = []
                    for ibank in rec_banks:

                        # index nonzero act units
                        r_tmp = recruit_ind[ibank]
                        act_nonzero = [act[ibank, recruit_ind[ibank]] != 0]
                        r_ind_tmp.append(r_tmp[act_nonzero])

                    # reassigning recruit ind
                    # - do i need recruit_ind or just flat is enough?
                    recruit_ind = r_ind_tmp
                    recruit_ind_flat = [item for sublist in recruit_ind
                                        for item in sublist]

                recruit_ind_flat = torch.tensor(recruit_ind_flat)

                # recruit n_mispredicted units
                if len(recruit_ind_flat):  # if none, skip
                    model.active_units[recruit_ind_flat] = True  # set ws to active
                    model.winning_units[:] = 0  # clear
                    model.winning_units[recruit_ind_flat] = True
                    # keep units that predicted correctly
                    if itrl > 0:
                        # for ibank in range(model.n_banks):
                        for i, ibank in enumerate(rec_banks):
                            not_mispred = (
                                torch.tensor(win_ind[ibank])[~mispred_units[i]]
                                )
                            model.winning_units[not_mispred] = True

                    model.units_pos[recruit_ind_flat] = x  # place at curr stim

                for ibank in rec_banks:
                    model.recruit_units_trl[ibank].append(itrl)

                # go through update again after cluster added
                optimizer.zero_grad()
                out, pr = model.forward(x)
                loss = criterion(out[0].unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                with torch.no_grad():
                    win_mask[:] = 0  # clear
                    win_mask[:, model.winning_units] = True  # update w winners
                    model.fc1.weight.grad.mul_(win_mask)
                    # diff learning rates per bank (set lr_nn to 1 above)
                    for ibank in range(model.n_banks):
                        model.fc1.weight.grad[:, model.bmask[ibank]] = (
                            model.fc1.weight.grad[:, model.bmask[ibank]]
                            * model.params['lr_nn'][ibank]
                            )

                model.attn.grad[:] = 0  # clear grad

                # remove nn updates for non-recruiting bank
                for ibank in upd_banks:
                    model.fc1.weight.grad[:, model.bmask[ibank].squeeze()] = 0

                optimizer.step()

                # save updated attn ws - save even if not update
                model.attn_trace.append(model.attn.detach().clone())

                # NOTE: TEMPORARILY COMMENTED OUT for gridsearch

                # update units pos w multiple banks - double update rule
                # - no need since on the stim - unless noise
                # for ibank in rec_banks:
                #     units_ind = (model.winning_units
                #                  & model.bmask[ibank].squeeze())
                #     update = (
                #         (x - model.units_pos[units_ind])
                #         * model.params['lr_clusters'][ibank]
                #         )
                #     model.units_pos[units_ind] += update

                #     # - step 2 - winners update towards self
                #     winner_mean = torch.mean(
                #         model.units_pos[units_ind], axis=0)
                #     update = (
                #         (winner_mean - model.units_pos[units_ind])
                #         * model.params['lr_clusters_group'][ibank])
                #     model.units_pos[units_ind] += update

                # save updated unit positions
                model.units_pos_trace.append(model.units_pos.detach().clone())

            itrl += 1

            # TMP removed
            # if torch.sum(model.fc1.weight == 0) == 0:  # no units to recruit
            #     warnings.warn("No more units to recruit")

        # save epoch acc (itrl needs to be -1, since it was updated above)
        epoch_acc[epoch] = trial_acc[itrl-len(inputs):itrl].mean()
        for ibank in range(model.n_banks + 1):
            epoch_ptarget[ibank, epoch] = (
                trial_ptarget[ibank, itrl-len(inputs):itrl].mean()
                )

    return model, epoch_acc, trial_acc, epoch_ptarget, trial_ptarget
