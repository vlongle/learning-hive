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
from shell.datasets.datasets import get_custom_tensordataset


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

    def process_communicate(self, task_id, communication_round, final=False):
        self.log(task_id, communication_round, info={'info': 'before'})
        self.aggregate_models()
        self.log(task_id, communication_round, info={'info': 'after'})


        if self.sharing_strategy.when_reoptimize_structure == 'never':
            return
        elif self.sharing_strategy.when_reoptimize_structure == 'final':
            if final:
                self.reoptimize_past_structures(task_id, communication_round)
        elif self.sharing_strategy.when_reoptimize_structure == 'always':
            self.reoptimize_past_structures(task_id, communication_round)
        else:
            raise NotImplementedError(f'when_reoptimize_structure: {self.sharing_strategy.when_reoptimize_structure} not implemented')


    def reoptimize_past_structures(self, task_id, communication_round, num_epochs=1):
        # optimize structures from 0--> task_id - 1 (excluding current task_id)
        # using the replay buffer
        # print("REOPTIMIZE TASK_ID:", task_id, len(self.agent.memory_loaders))
        assert len(self.agent.memory_loaders.values()) == task_id, f"task_id: {task_id}, len(self.agent.memory_loaders.values()): {len(self.agent.memory_loaders.values())}"

        # TODO: handle the component dropout carefully. 
        # print("BEFORE CURRENT_ACTIVE_CANDIDATE_INDEX", self.net.active_candidate_index)
        current_active_candidate_index = self.net.active_candidate_index
        self.net.active_candidate_index = None

        # NOTE: at the final communication round for the task, we need to reset the adam optimizer.
        # Adam uses a moving average of the past gradients, so if the model dynamically changes (e.g.,
        # we remove the conditional component), we will get an error.
        # This wasn't a problem before for some reasons, although re-optimizing past modules (using experience
        # replay) seems like it should cause the same problem.
        # if communication_round == self.num_coms[task_id]-1:
        if task_id not in self.num_coms: 
        # HACK: hacky way to identify the final communication round basically
        # for the task. The task_id will be +1 the actual task because
        # we've completed the last training epoch (communication happens AFTER
        # training). So the current task is "just" considered past.
            # print("RESETING OPTIMIZER")
            self.agent.optimizer = torch.optim.Adam(self.net.parameters(),)
        # print("STRUCTURE", self.net.structure)
        for _ in range(num_epochs):
            for task in range(task_id):
                self.net.freeze_modules()
                self.net.freeze_structure()
                self.net.unfreeze_structure(task_id=task)

                # update structure for task
                loader = torch.utils.data.DataLoader(get_custom_tensordataset(self.agent.memory_loaders[task].dataset.tensors,
                                                  name=self.agent.dataset_name,
                                      use_contrastive=self.agent.use_contrastive),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True,
                )

                for X, Y, _ in loader:
                    if isinstance(X, list):
                        # contrastive two views
                        X = torch.cat([X[0], X[1]], dim=0)
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    # with new module
                    # print(self.net.structure)
                    self.agent.update_structure(
                        X, Y, task)
        self.net.active_candidate_index = current_active_candidate_index
         


@ray.remote
class ParallelModGrad(ModGrad):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors:
            ray.get(neighbor.receive.remote(self.node_id, deepcopy(self.model), "model"))