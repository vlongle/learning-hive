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
        # elif self.sharing_strategy.when_reoptimize_structure == 'final':
        #     if final:
        #         self.reoptimize_past_structures(task_id, communication_round)
        # elif self.sharing_strategy.when_reoptimize_structure == 'always':
        #     self.reoptimize_past_structures(task_id, communication_round)
        else:
            raise NotImplementedError(
                f'when_reoptimize_structure: {self.sharing_strategy.when_reoptimize_structure} not implemented')


@ray.remote
class ParallelModGrad(ModGrad):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
