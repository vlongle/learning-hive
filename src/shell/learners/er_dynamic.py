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
import pandas as pd


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels,
        indices: list = None,
        num_samples: int = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))
                            ) if indices is None else indices
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(
            self.indices) if num_samples is None else num_samples
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class CompositionalDynamicER(CompositionalDynamicLearner):
    def __init__(self, net, memory_size, save_dir='./tmp/results/',
                 improvement_threshold=0.05, use_contrastive=False, dataset_name=None,
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
                         delta_ood=delta_ood,)
        self.replay_buffers = {}
        # self.memory_loaders = {}
        self.memory_size = memory_size

    def update_modules(self, trainloader, task_id, train_mode=None, global_step=None,
                       use_aux=True):
        """
        NOTE: for contrastive, during accommodation,
        we should also allow past decoders to change so that gradients can flow
        to CE.
        """
        self.net.unfreeze_modules()
        self.net.freeze_structure()

        prev_reduction = self.get_loss_reduction()
        self.set_loss_reduction('sum')

        tmp_dataset = copy.deepcopy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + \
            (torch.full((len(tmp_dataset),), task_id, dtype=int),)

        mega_dataset = ConcatDataset(
            [get_custom_tensordataset(replay.get_tensors(), name=self.dataset_name,
                                      use_contrastive=self.use_contrastive) for t, replay in self.replay_buffers.items()] + [tmp_dataset]
            + [get_custom_tensordataset(replay.get_tensors(), name=self.dataset_name,
                                        use_contrastive=self.use_contrastive) for t, replay in self.shared_replay_buffers.items()
               if t != task_id and len(replay) > 0]
        )

        batch_size = trainloader.batch_size

        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  pin_memory=True
                                                  )

        # NOTE: this might be buggy?
        for module_idx in self.net.candidate_indices:
            # Select the module to be used in this round
            self.net.select_active_module(module_idx)
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
                    # for task_id_tmp in sorted(all_t.tolist(), reverse=True):
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
                                           mode=train_mode,
                                           global_step=global_step,
                                           use_aux=use_aux,
                                           )
                n = len(Y)
                l /= n
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                l = 0.
                n = 0
                self.net.hide_tmp_modulev2()
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
                                           mode=train_mode,
                                           global_step=global_step,
                                           use_aux=use_aux,
                                           )
                n = len(Y)
                l /= n
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                self.net.recover_hidden_modulev2()

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
                X, _ = X
            self.replay_buffers[task_id].push(X, Y)

        # self.memory_loaders[task_id] = (
        #     torch.utils.data.DataLoader(self.replay_buffers[task_id],
        #                                 batch_size=trainloader.batch_size,
        #                                 shuffle=True,
        #                                 num_workers=2,
        #                                 pin_memory=True
        #                                 ))
