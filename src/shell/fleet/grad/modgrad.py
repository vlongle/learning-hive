'''
File: /modgrad.py
Project: shell
Created Date: Monday March 27th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import logging
from copy import deepcopy
import ray
from shell.fleet.grad.monograd import ModelSyncAgent, ParallelModelSyncAgent
import copy
import torch
from torch.utils.data.dataset import ConcatDataset


class ModGrad(ModelSyncAgent):
    def prepare_model(self):
        # NOTE: we only share the initial components, so we remove
        # anything more than that
        num_init_components = self.net.depth
        # add to self.excluded_params
        num_components = len(self.net.components)
        for i in range(num_init_components, num_components):
            self.excluded_params.add("components.{}".format(i))
        return super().prepare_model()

    def process_communicate(self, task_id, communication_round):
        self.aggregate_models()
        # ModGrad: retrain on local tasks using experience replay. Update ONLY shared modules,
        # keeping structures and other task-specific modules fixed.
        trainloader = (
            torch.utils.data.DataLoader(self.dataset.trainset[task_id],
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        ))

        self.finetune_shared_modules(trainloader, task_id)

    def finetune_shared_modules(self, trainloader, task_id, train_mode=None):
        self.net.freeze_structure()

        # freeze all the modules except the shared ones and the task decoder.
        self.net.freeze_modules()
        self.net.unfreeze_some_modules(range(self.net.depth))
        self.net.unfreeze_decoder(task_id)

        # training
        prev_reduction = self.agent.get_loss_reduction()
        self.agent.set_loss_reduction('sum')

        tmp_dataset = copy.copy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + \
            (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        mega_dataset = ConcatDataset(
            [loader.dataset for loader in self.agent.memory_loaders.values()] + [tmp_dataset])
        batch_size = trainloader.batch_size
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
                l += self.agent.compute_loss(X[t == task_id_tmp],
                                             Y[t == task_id_tmp], task_id_tmp,
                                             mode=train_mode)
                n += X.shape[0]
            l /= n
            self.agent.optimizer.zero_grad()
            l.backward()
            self.agent.optimizer.step()

        # undo all the freezing stuff
        self.agent.set_loss_reduction(prev_reduction)
        self.net.unfreeze_structure(task_id)
        self.net.freeze_modules()


@ray.remote
class ParallelModGrad(ModGrad):
    def communicate(self, task_id, communication_round):
        # logging.info(
        #     f"node {self.node_id} is communicating at round {communication_round} for task {task_id}")
        # TODO: Should we do deepcopy???
        # put model on object store
        # state_dict = deepcopy(self.net.state_dict())
        # model = state_dict
        # model = ray.put(state_dict)
        # send model to neighbors
        # logging.info(f"My neighbors are: {self.neighbors}")
        for neighbor in self.neighbors:
            # neighbor_id = ray.get(neighbor.get_node_id.remote())
            # NOTE: neighbor_id for some reason is NOT responding...
            # logging.info(f"SENDING MODEL: {self.node_id} -> {neighbor_id}")
            ray.get(neighbor.receive.remote(self.node_id, self.model, "model"))
            self.bytes_sent[(task_id, communication_round)
                            ] = self.compute_model_size(self.model)
