'''
File: /er_nocomponents.py
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
from shell.learners.base_learning_classes import Learner


class NoComponentsER(Learner):
    def __init__(self, net, memory_size, save_dir='./tmp/results/',  improvement_threshold=0.05):
        super().__init__(net, save_dir,  improvement_threshold=improvement_threshold)
        self.replay_buffers = {}
        self.memory_loaders = {}
        self.memory_size = memory_size

    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None, valloader=None,
              eval_bool=True):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        self.save_data(0, task_id, testloaders)  # zeroshot eval
        if self.T <= self.net.num_init_tasks:
            self.init_train(trainloader, task_id, num_epochs,
                            save_freq, testloaders)
        else:

            tmp_dataset = copy.copy(trainloader.dataset)
            tmp_dataset.tensors = tmp_dataset.tensors + \
                (torch.full((len(tmp_dataset),), task_id, dtype=int),)
            mega_dataset = ConcatDataset(
                [loader.dataset for loader in self.memory_loaders.values()] + [tmp_dataset])
            mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                      batch_size=trainloader.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True
                                                      )
            self._train(mega_loader, num_epochs, task_id,
                        testloaders, save_freq, eval_bool)
            self.save_data(num_epochs + 1, task_id,
                           testloaders, final_save=True)  # final eval
            self.update_multitask_cost(trainloader, task_id)

    def _train(self, mega_loader, num_epochs, task_id, testloaders, save_freq=1, eval_bool=True):
        prev_reduction = self.loss.reduction
        self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        for i in range(num_epochs):
            for X, Y, t in mega_loader:
                X = X.to(self.net.device, non_blocking=True)
                Y = Y.to(self.net.device, non_blocking=True)
                l = 0.
                n = 0
                all_t = torch.unique(t)
                for task_id_tmp in all_t:
                    Y_hat = self.net(X[t == task_id_tmp],
                                     task_id=task_id_tmp)
                    l += self.loss(Y_hat, Y[t == task_id_tmp])
                    n += X.shape[0]
                l /= n
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
            if i % save_freq == 0 or i == num_epochs - 1:
                if eval_bool:
                    self.evaluate(testloaders)
                self.save_data(i + 1, task_id, testloaders)
        self.loss.reduction = prev_reduction

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
