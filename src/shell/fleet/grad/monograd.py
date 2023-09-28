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


class ModelSyncAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
        self.incoming_models = {}
        # key: (task_id, communication_round, neighbor_id)
        self.bytes_sent = {}
        # self.num_init_tasks = net_kwargs.get("num_init_tasks", 1)
        self.excluded_params = set(
            ["decoder", "projector", "structure"])

        self.sharing_record = Record(os.path.join(
            self.save_dir,
            "sharing_record.csv"
        ))
        self.retrain_record = Record(os.path.join(
            self.save_dir, "retrain_record.csv"))

    def compute_model_size(self, state_dict):
        return sum(p.numel() for p in state_dict.values())

    def prepare_model(self):
        # return self.net.state_dict()
        # without the task-specific parameters named in self.excluded_params
        # should also exclude random_linear_projection
        self.excluded_params.add("random_linear_projection")
        model = exclude_model(self.net.state_dict(), self.excluded_params)

        return model

    def prepare_communicate(self, task_id, communication_round):
        self.model = self.prepare_model()

    def communicate(self, task_id, communication_round):
        # if task_id < self.num_init_tasks - 1:
        #     return
        # send model to neighbors
        for neighbor in self.neighbors:
            neighbor.receive(self.node_id, deepcopy(self.model), "model")
            self.bytes_sent[(task_id, communication_round,
                             neighbor.get_node_id())] = self.compute_model_size(self.model)

    def receive(self, node_id, model, msg_type):
        # get model from neighbors
        # average all the models together!
        self.incoming_models[node_id] = model

    def get_received_models(self):
        return self.incoming_models

    def get_bytes_sent(self):
        return self.bytes_sent

    def log(self, task_id, communication_round):
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

        self.sharing_record.write(
            {
                "task_id": task_id,
                "communication_round": communication_round,
            } | diffs['avg_neigh']
        )
        self.sharing_record.save()

    def aggregate_models(self):
        # get model from neighbors
        # average all the models together!
        logging.info("AGGREGATING MODELS...")
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

    def retrain(self, num_epochs, task_id, testloaders, save_freq=1, eval_bool=True):
        """
        Retrain on local data after aggregation. Only tested for monolithic models.
        """
        # mega_dataset = ConcatDataset(
        #     [loader.dataset for loader in self.agent.memory_loaders.values()])

        mega_dataset = ConcatDataset(
            [get_custom_tensordataset(loader.dataset.tensors, name=self.agent.dataset_name,
                                      use_contrastive=self.agent.use_contrastive) for loader in self.agent.memory_loaders.values()])
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=self.agent.memory_loaders[0].batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True
                                                  )
        logging.info("RETRAINING...")
        self.agent._train(mega_loader, num_epochs, task_id,
                          testloaders, save_freq, eval_bool,
                          record=self.retrain_record)

    def process_communicate(self, task_id, communication_round):
        if communication_round % self.sharing_strategy.log_freq == 0:
            self.log(task_id, communication_round)
        self.aggregate_models()

        # # Monograd: retrain on local tasks using experience replay
        # testloaders = {task: torch.utils.data.DataLoader(testset,
        #                                                  batch_size=128,
        #                                                  shuffle=False,
        #                                                  num_workers=0,
        #                                                  pin_memory=True,
        #                                                  ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}

        # self.agent.save_data(0,
        #                      task_id, testloaders, final_save=False,
        #                      record=self.retrain_record)  # zeroshot
        # # TODO: the epoch here is not incremented with respect to the previous communication round.
        # self.retrain(
        #     self.sharing_strategy.retrain.num_epochs, task_id, testloaders, save_freq=1, eval_bool=True,
        # )

        # self.agent.save_data(self.sharing_strategy.retrain.num_epochs+1,
        #                      task_id, testloaders, final_save=True,
        #                      record=self.retrain_record,
        #                      save_dir=os.path.join(
        #                          self.agent.save_dir, "retrain"),
        #                      )  # save final model

    def replace_model(self, new_model, strict=True):
        # print("replacing model with strict:", strict)
        self.net.load_state_dict(new_model, strict=strict)
        self.net.to(self.net.device)


@ray.remote
class ParallelModelSyncAgent(ModelSyncAgent):
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
            # use ray.get blocking to make sure that the receiver has received the model
            ray.get(neighbor.receive.remote(self.node_id, self.model, "model"))
            self.bytes_sent[(task_id, communication_round)
                            ] = self.compute_model_size(self.model)
