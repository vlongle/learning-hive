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
        num_init_components = self.net.depth
        num_components = len(self.net.components)
        for i in range(num_init_components, num_components):
            self.excluded_params.add("components.{}".format(i))
        return super().prepare_model()


@ray.remote
class ParallelModGrad(ModGrad):
    def communicate(self, task_id, communication_round, final=False, strategy=None):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
