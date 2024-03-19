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
from shell.datasets.datasets import get_custom_tensordataset


class CompositionalDynamicER(CompositionalDynamicLearner):
    def __init__(self, net, memory_size, save_dir='./tmp/results/',
                 improvement_threshold=0.05, use_contrastive=False, dataset_name=None):
        super().__init__(net, save_dir,  improvement_threshold=improvement_threshold,
                         use_contrastive=use_contrastive, dataset_name=dataset_name)
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
            [get_custom_tensordataset(loader.dataset.tensors, name=self.dataset_name,
                                      use_contrastive=self.use_contrastive) for loader in self.memory_loaders.values()] + [tmp_dataset])
        # tmp_loader = next(iter(self.memory_loaders.values()))
        # batch_size = tmp_loader.batch_size
        # NOTE: changed to account for cases when we don't
        # have any init tasks i.e. len(self.memory_loaders) == 0
        batch_size = trainloader.batch_size
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True
                                                  )
        for X, Y, t in mega_loader:
            if isinstance(X, list):
                # contrastive two views
                X = torch.cat([X[0], X[1]], dim=0)
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            l = 0.
            n = 0
            all_t = torch.unique(t)

            if self.use_contrastive:
                Xhaf = X[:len(X)//2]
                Xother = X[len(X)//2:]

            for task_id_tmp in all_t:
                Yt = Y[t == task_id_tmp]
                if self.use_contrastive:
                    # Xt will be twice as long as Yt
                    # use advanced indexing to get the first half
                    Xt_haf = Xhaf[t == task_id_tmp]
                    Xt_other = Xother[t == task_id_tmp]
                    Xt = torch.cat([Xt_haf, Xt_other], dim=0)
                else:
                    Xt = X[t == task_id_tmp]
                l += self.compute_loss(Xt,
                                       Yt, task_id_tmp,
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
                Yt = Y[t == task_id_tmp]
                if self.use_contrastive:
                    # Xt will be twice as long as Yt
                    # use advanced indexing to get the first half
                    Xt_haf = Xhaf[t == task_id_tmp]
                    Xt_other = Xother[t == task_id_tmp]
                    Xt = torch.cat([Xt_haf, Xt_other], dim=0)
                else:
                    Xt = X[t == task_id_tmp]
                l += self.compute_loss(Xt,
                                       Yt, task_id_tmp,
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
            if isinstance(X, list):
                # contrastive two views
                X = X[0]  # only store the first view (original image)
            self.replay_buffers[task_id].push(X, Y)
        # self.memory_loaders[task_id] = (
        #     torch.utils.data.DataLoader(self.replay_buffers[task_id],
        #                                 batch_size=trainloader.batch_size,
        #                                 shuffle=True,
        #                                 num_workers=10,
        #                                 pin_memory=True
        #                                 ))
