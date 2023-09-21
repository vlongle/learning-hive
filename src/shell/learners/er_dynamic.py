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
                 improvement_threshold=0.05, use_contrastive=False, dataset_name=None):
        super().__init__(net, save_dir,  improvement_threshold=improvement_threshold,
                         use_contrastive=use_contrastive, dataset_name=dataset_name)
        self.replay_buffers = {}
        self.aug_replay_buffers = {}
        self.memory_loaders = {}
        self.memory_size = memory_size

    def update_modules(self, trainloader, task_id, train_mode=None):
        """
        NOTE: for contrastive, during accommodation,
        we should also allow past decoders to change so that gradients can flow
        to CE.
        """
        self.net.unfreeze_modules()
        # self.net.freeze_structure(freeze=True)
        self.net.freeze_structure()

        # NEW: ====================
        # if self.use_contrastive:
        #     for t in range(task_id+1):
        #         self.net.unfreeze_decoder(t)
        # =========================

        # prev_reduction = self.loss.reduction
        # self.loss.reduction = 'sum'     # make sure the loss is summed over instances

        # if self.use_contrastive:
        #     for t in range(task_id+1):
        #         # unfreeze projection as well
        #         self.net.unfreeze_projector(t)
        # if self.use_contrastive:
        #     for t in range(task_id+1):
        #         # unfreeze projection as well
        #         self.net.unfreeze_projector(t)
        #     # for t in range(task_id+1):
        #     #     self.net.unfreeze_decoder(t)

        prev_reduction = self.get_loss_reduction()
        self.set_loss_reduction('sum')

        # # NEW: ====================
        # if self.use_contrastive:
        #     for t in range(task_id+1):
        #         self.net.unfreeze_decoder(t)
        # =========================

        # tmp_dataset = copy.copy(trainloader.dataset)
        tmp_dataset = copy.deepcopy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + \
            (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        # if self.use_contrastive:
        #     def combine_datasets(tensors, aug_tensors):
        #         # tensors: (X, y, t)
        #         # aug_tensors: (X_aug, y, t)
        #         # returns: ([X, X_aug], y, t)
        #         return (torch.cat((tensors[0], aug_tensors[0])), tensors[1], tensors[2])
        #     mega_dataset = ConcatDataset(
        #         [combine_datasets(dataset.tensors, dataset_aug.tensors) for dataset, dataset_aug in zip(
        #             self.memory_loaders.values(), self.memory_loaders_aug.values()
        #         )] + [tmp_dataset])
        # else:
        #     mega_dataset = ConcatDataset(
        #         [get_custom_tensordataset(loader.dataset.tensors, name=self.dataset_name,
        #                                   use_contrastive=self.use_contrastive) for loader in self.memory_loaders.values()] + [tmp_dataset])

        # from torch.utils.data import TensorDataset

        # class CustomTensorDataset2(TensorDataset):
        #     # tensordataset but also apply transforms
        #     def __init__(self, *tensors, X_aug):
        #         super().__init__(*tensors)
        #         self.X_aug = X_aug

        #     def __getitem__(self, index):
        #         tensors = super().__getitem__(index)
        #         x = tensors[0]
        #         # if self.transform, apply it on the first tensor
        #         x_aug = self.X_aug[index]
        #         x = [x, x_aug]
        #         return tuple([x] + list(tensors[1:]))

        mega_dataset = ConcatDataset(
            [get_custom_tensordataset(loader.dataset.tensors, name=self.dataset_name,
                                      use_contrastive=self.use_contrastive) for loader in self.memory_loaders.values()] + [tmp_dataset])
        print('dataset:', len(mega_dataset))
        # mega_dataset = ConcatDataset(
        #     [CustomTensorDataset2(*dataset.tensors,
        #                           X_aug=dataset2.tensors[0]) for dataset, dataset2 in zip(
        #         self.replay_buffers.values(), self.aug_replay_buffers.values()
        #     )] + [tmp_dataset])
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

        # if self.use_contrastive:
        #     # ensure that each task is equally represented in a batch
        #     # because contrastive supCon/SimCLR is sensitive to the number
        #     # of contrast pairs, and we don't want the current task number of
        #     # contrast pairs to be too large compared to the previous tasks, leading
        #     # to loss dominance by the current task
        #     labels = torch.cat([loader.dataset.tensors[2]
        #                        for loader in self.memory_loaders.values()] + [tmp_dataset.tensors[2]])
        #     mega_loader = torch.utils.data.DataLoader(
        #         mega_dataset,
        #         sampler=ImbalancedDatasetSampler(mega_dataset, labels),
        #         batch_size=batch_size,
        #     )
        # else:
        #     mega_loader = torch.utils.data.DataLoader(mega_dataset,
        #                                               batch_size=batch_size,
        #                                               shuffle=True,
        #                                               num_workers=0,
        #                                               pin_memory=True
        #                                               )
        # print("len(loader):", len(mega_loader))
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
                                       )
                # n += X.shape[0]
            n = len(Y)
            l /= n
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            l = 0.
            n = 0
            # self.net.hide_tmp_module()
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
                                       mode=train_mode)
                # n += X.shape[0]
            n = len(Y)
            l /= n
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            # self.net.recover_hidden_module()
            self.net.recover_hidden_modulev2()

        # NEW: ====================
        # if self.use_contrastive:
        #     self.net.freeze_decoder()
        # =========================

        # self.loss.reduction = prev_reduction
        self.set_loss_reduction(prev_reduction)
        self.net.freeze_modules()
        # unfreeze only current task's structure
        # self.net.freeze_structure(freeze=False, task_id=task_id)
        self.net.unfreeze_structure(task_id)

    def update_multitask_cost(self, trainloader, task_id):
        self.replay_buffers[task_id] = ReplayBufferReservoir(
            self.memory_size, task_id)
        if self.use_contrastive:
            self.aug_replay_buffers[task_id] = ReplayBufferReservoir(
                self.memory_size, task_id)
        for X, Y in trainloader:
            if isinstance(X, list):
                # contrastive two views
                # X = X[0]  # only store the first view (original image)
                # X_aug = X[1]
                X, X_aug = X
                self.aug_replay_buffers[task_id].push(X_aug, Y)

            self.replay_buffers[task_id].push(X, Y)
        self.memory_loaders[task_id] = (
            torch.utils.data.DataLoader(self.replay_buffers[task_id],
                                        batch_size=trainloader.batch_size,
                                        shuffle=True,
                                        num_workers=10,
                                        pin_memory=True
                                        ))
