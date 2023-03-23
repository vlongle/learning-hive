'''
File: /er_dynamic.py
Project: learners
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import torch
import torch.nn as nn
import os
from torch.utils.data.dataset import ConcatDataset
import copy
from shell.utils.replay_buffers import ReplayBufferReservoir
from shell.learners.base_learning_classes import CompositionalDynamicLearner


class CompositionalDynamicER(CompositionalDynamicLearner):
    def __init__(self, net, memory_size, save_dir='./tmp/results/',
                 improvement_threshold=0.05, use_contrastive=False):
        super().__init__(net, save_dir,  improvement_threshold=improvement_threshold,
                         use_contrastive=use_contrastive)
        self.replay_buffers = {}
        self.memory_loaders = {}
        self.memory_size = memory_size

    def update_modules(self, trainloader, task_id, train_mode=None):
        self.net.unfreeze_modules()
        # self.net.freeze_structure(freeze=True)
        self.net.freeze_structure()
        # prev_reduction = self.loss.reduction
        # self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        prev_reduction = self.get_loss_reduction()
        self.set_loss_reduction('sum')

        tmp_dataset = copy.copy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + \
            (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        mega_dataset = ConcatDataset(
            [loader.dataset for loader in self.memory_loaders.values()] + [tmp_dataset])
        tmp_loader = next(iter(self.memory_loaders.values()))
        batch_size = tmp_loader.batch_size
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True
                                                  )
        for X, Y, t in mega_loader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            l = 0.
            n = 0
            all_t = torch.unique(t)
            for task_id_tmp in all_t:
                # Y_hat = self.net(X[t == task_id_tmp], task_id=task_id_tmp)
                # l += self.loss(Y_hat, Y[t == task_id_tmp])
                l += self.compute_loss(X[t == task_id_tmp],
                                       Y[t == task_id_tmp], task_id_tmp,
                                       mode=train_mode)
                n += X.shape[0]
            l /= n
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            l = 0.
            n = 0
            self.net.hide_tmp_module()
            for task_id_tmp in all_t:
                # Y_hat = self.net(X[t == task_id_tmp], task_id=task_id_tmp)
                # l += self.loss(Y_hat, Y[t == task_id_tmp])
                l += self.compute_loss(X[t == task_id_tmp],
                                       Y[t == task_id_tmp], task_id_tmp,
                                       mode=train_mode)
                n += X.shape[0]
            l /= n
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            self.net.recover_hidden_module()

        # self.loss.reduction = prev_reduction
        self.set_loss_reduction(prev_reduction)
        self.net.freeze_modules()
        # unfreeze only current task's structure
        # self.net.freeze_structure(freeze=False, task_id=task_id)
        self.net.unfreeze_structure(task_id)

    def update_multitask_cost(self, trainloader, task_id):
        self.replay_buffers[task_id] = ReplayBufferReservoir(
            self.memory_size, task_id)
        for X, Y in trainloader:
            self.replay_buffers[task_id].push(X, Y)
        self.memory_loaders[task_id] = (
            torch.utils.data.DataLoader(self.replay_buffers[task_id],
                                        batch_size=trainloader.batch_size,
                                        shuffle=True,
                                        num_workers=10,
                                        pin_memory=True
                                        ))
