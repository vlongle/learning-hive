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
            ["decoder", "structure", "projector"])

        self.sharing_model_diff_record = Record(os.path.join(
            self.save_dir,
            "sharing_record.csv"
        ))
        self.sharing_perf_record = Record(os.path.join(
            self.save_dir,
            "sharing_perf_record.csv"
        ))

    def prepare_communicate(self, task_id, end_epoch, comm_freq, num_epochs,
                            communication_round, final=False):
        self.model = exclude_model(
            deepcopy(self.net.state_dict()), self.excluded_params)

    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            neighbor.receive(self.node_id, deepcopy(self.model), "model")

    def receive(self, node_id, model, msg_type):
        self.incoming_models[node_id] = model

    def log(self, task_id, communication_round, info={}):
        self.log_model_diff(task_id, communication_round, info)
        # self.log_model_perf(task_id, communication_round, info)

    # def log_model_perf(self, task_id, communication_round, info={}):
    #     testloaders = {task: torch.utils.data.DataLoader(testset,
    #                                                      batch_size=256,
    #                                                      shuffle=False,
    #                                                      num_workers=4,
    #                                                      pin_memory=True,
    #                                                      ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
    #     _, test_acc = self.agent.evaluate(testloaders) # test_acc is a dict of task: acc
    #     for t, t_a in test_acc.items():
    #         if isinstance(t_a, tuple):
    #             test_acc[t] = max(t_a)

    #     if "avg" not in test_acc:
    #         test_acc["avg"] = sum(
    #             test_acc.values()) / len(test_acc)
    #     test_acc_ls = [{"test_task": test_task_id, "test_acc": t_a} for test_task_id, t_a in test_acc.items()]
    #     # make a test_acc_ls of dicts where each entry is {task_id: , test_acc:}
    #     for entry in test_acc_ls:
    #         self.sharing_perf_record.write(
    #             {
    #                 "task_id": task_id,
    #                 "communication_round": communication_round,
    #             } |entry | info
    #         )
    #     self.sharing_perf_record.save()

    def log_model_diff(self, task_id, communication_round, info={}):
        my_model = self.net.state_dict()
        diffs = {}  # diff['name'] is a dictionary of "param_key"
        # and float indicating the difference
        for neigh, inc_model in self.incoming_models.items():
            diffs[neigh] = diff_models(my_model, inc_model)
        # compute diff['avg'] which is a dictionary of "param_key"
        # and float where the value is averaged over all the neighbors
        # in diffs
        diffs['avg_neigh'] = {}
        for name, diff in diffs.items():
            for param_key, value in diff.items():
                if param_key not in diffs['avg_neigh']:
                    diffs['avg_neigh'][param_key] = 0
                diffs['avg_neigh'][param_key] += value

        for param_key, value in diffs['avg_neigh'].items():
            diffs['avg_neigh'][param_key] = value / len(self.incoming_models)

        # diffs['avg_neigh']['avg_params'] is averaged over all param_key
        diffs['avg_neigh']['avg_params'] = sum(
            diffs['avg_neigh'].values()) / len(diffs['avg_neigh'])

        self.sharing_model_diff_record.write(
            {
                "task_id": task_id,
                "communication_round": communication_round,
            } | diffs['avg_neigh'] | info
        )
        self.sharing_model_diff_record.save()

    def aggregate_models(self):
        # get model from neighbors
        # average all the models together!
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
            # self.bytes_sent[(task_id, communication_round)
            #                 ] = self.compute_model_size(self.model)
