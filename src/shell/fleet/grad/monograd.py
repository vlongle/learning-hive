'''
File: /gradient_fleet.py
Project: fleet
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import logging
import ray
from shell.fleet.fleet import Agent
from copy import deepcopy
import torch
from torch.utils.data.dataset import ConcatDataset
from shell.fleet.utils.model_sharing_utils import exclude_model, diff_models
from collections import defaultdict
from shell.utils.record import Record
from shell.datasets.datasets import get_custom_tensordataset
import os


"""
Only sync modules
"""


class ModelSyncAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
        self.incoming_models = {}
        self.excluded_params = set(
            ["decoder", "structure", "projector", "random_linear_projection"])

        self.sharing_model_diff_record = Record(os.path.join(
            self.save_dir,
            "sharing_record.csv"
        ))
        self.sharing_perf_record = Record(os.path.join(
            self.save_dir,
            "sharing_perf_record.csv"
        ))

    def prepare_model(self):
        return exclude_model(
            deepcopy(self.net.state_dict()), self.excluded_params)

    def prepare_communicate(self, task_id, end_epoch, comm_freq, num_epochs,
                            communication_round, final=False):
        self.model = self.prepare_model()

    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            neighbor.receive(self.node_id, deepcopy(self.model), "model")

    def receive(self, node_id, model, msg_type):
        self.incoming_models[node_id] = model

    def log(self, task_id, communication_round, info={}):
        self.log_model_diff(task_id, communication_round, info)
        # self.log_model_perf(task_id, communication_round, info)

    def log_model_diff(self, task_id, communication_round, info={}):
        my_model = self.net.state_dict()
        diffs = {neigh: diff_models(my_model, inc_model)
                 for neigh, inc_model in self.incoming_models.items()}

        # Calculate average difference per parameter across all models
        avg_diffs = self.calculate_average_diffs(diffs)
        if len(avg_diffs) == 0:
            return

        # Calculate overall average of these differences
        avg_diffs['avg_params'] = sum(avg_diffs.values()) / len(avg_diffs)

        # Record the diffs with task info
        record = {"task_id": task_id,
                  "communication_round": communication_round, **avg_diffs, **info}

        self.sharing_model_diff_record.write(record)
        self.sharing_model_diff_record.save()

    def calculate_average_diffs(self, diffs):
        param_keys = set(key for diff in diffs.values() for key in diff)
        avg_diffs = {key: sum(diff.get(
            key, 0) for diff in diffs.values()) / len(diffs) for key in param_keys}
        return avg_diffs

    def aggregate_models(self):
        # get model from neighbors
        # average all the models together!
        if len(self.incoming_models.values()) == 0:
            return
        logging.info("AGGREGATING MODELS...no_components %s",
                     len(self.net.components))
        stuff_added = defaultdict(int)
        for model in self.incoming_models.values():
            for name, param in model.items():
                # print("Adding name:", name)
                self.net.state_dict()[name].data += param.data
                stuff_added[name] += 1

        # normalize
        for name, param in self.net.state_dict().items():
            # +1 because it includes the current model
            param.data /= stuff_added[name] + 1

    def process_communicate(self, task_id, communication_round, final=False):
        self.log(task_id, communication_round, info={'info': 'before'})
        self.aggregate_models()
        self.log(task_id, communication_round, info={'info': 'after'})


@ray.remote
class ParallelModelSyncAgent(ModelSyncAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
