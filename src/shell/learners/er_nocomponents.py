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
from shell.datasets.datasets import get_custom_tensordataset


class NoComponentsER(Learner):
    def __init__(self, net, memory_size, save_dir='./tmp/results/',  improvement_threshold=0.05,
                 use_contrastive=False, dataset_name=None,
                 fl_strategy=None,
                 mu=None,
                 use_ood_separation_loss=False,
                 lambda_ood=2.0,
                 delta_ood=1.0,):
        super().__init__(net, save_dir,  improvement_threshold=improvement_threshold,
                         use_contrastive=use_contrastive, dataset_name=dataset_name,
                         fl_strategy=fl_strategy,
                         mu=mu,
                         use_ood_separation_loss=use_ood_separation_loss,
                         lambda_ood=lambda_ood,
                         delta_ood=delta_ood,
                         )
        self.replay_buffers = {}
        self.shared_replay_buffers = {}  # received from neighbors
        # self.memory_loaders = {}
        self.memory_size = memory_size

    def train(self, trainloader, task_id, component_update_freq=100,
              start_epoch=0, num_epochs=100, save_freq=1, testloaders=None, valloader=None,
              eval_bool=True, train_mode=None,
              record=None, final=True):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        if start_epoch == 0:
            self.save_data(start_epoch, task_id, testloaders,
                           mode=train_mode,
                           record=record)  # zeroshot eval
        # INIT TRAINING
        if self.T <= self.net.num_init_tasks:
            self.init_train(trainloader, task_id, start_epoch, num_epochs,
                            save_freq, testloaders)
        # CONTINUAL TRAINING
        else:
            # assume that trainloader.dataset is already a customTensorDataset
            tmp_dataset = copy.copy(trainloader.dataset)
            tmp_dataset.tensors = tmp_dataset.tensors + \
                (torch.full((len(tmp_dataset),), task_id, dtype=int),)
            # mega_dataset = ConcatDataset(
            # [get_custom_tensordataset(loader.dataset.tensors, name=self.dataset_name,
            #                           use_contrastive=self.use_contrastive) for loader in self.memory_loaders.values()] + [tmp_dataset])

            # self.make_shared_memory_loaders(batch_size=trainloader.batch_size)

            mega_dataset = ConcatDataset(
                [get_custom_tensordataset(replay.get_tensors(), name=self.dataset_name,
                                          use_contrastive=self.use_contrastive) for replay in self.replay_buffers.values()] + [tmp_dataset]
                + [get_custom_tensordataset(replay.get_tensors(), name=self.dataset_name,
                                            use_contrastive=self.use_contrastive) for replay in self.shared_replay_buffers.values()]
            )
            mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                      batch_size=trainloader.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      pin_memory=True
                                                      )
            self._train(mega_loader, start_epoch, num_epochs, task_id,
                        testloaders, save_freq, eval_bool, train_mode=train_mode)
            if final:
                self.save_data(num_epochs + start_epoch + 1, task_id,
                               testloaders, final_save=True, mode=train_mode,
                               record=record)  # final eval
                self.update_multitask_cost(trainloader, task_id)

    def _train(self, mega_loader, start_epoch, num_epochs, task_id, testloaders, save_freq=1, eval_bool=True,
               train_mode=None,
               record=None):
        # prev_reduction = self.loss.reduction
        # self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        prev_reduction = self.get_loss_reduction()
        self.set_loss_reduction('sum')
        for i in range(start_epoch, num_epochs + start_epoch):
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
                    # Y_hat = self.net(X[t == task_id_tmp],
                    #                  task_id=task_id_tmp)
                    # l += self.loss(Y_hat, Y[t == task_id_tmp])
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
                                           Yt,
                                           task_id_tmp, mode=train_mode,
                                           global_step=i,)
                    # n += X.shape[0]
                n = len(Y)
                l /= n
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
            if i % save_freq == 0 or i == num_epochs - 1:
                self.save_data(i + 1, task_id, testloaders, mode=train_mode,
                               record=record)
        # self.loss.reduction = prev_reduction
        self.set_loss_reduction(prev_reduction)

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
        #                                 num_workers=2,
        #                                 pin_memory=True
        #                                 ))
