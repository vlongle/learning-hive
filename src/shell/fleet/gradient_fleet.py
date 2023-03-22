'''
File: /gradient_fleet.py
Project: fleet
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import ray
from shell.fleet.fleet import Agent
from copy import deepcopy
import torch
from torch.utils.data.dataset import ConcatDataset


class ModelSyncAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs)
        self.models = {}
        # key: (task_id, communication_round, neighbor_id)
        self.bytes_sent = {}
        self.num_init_tasks = net_kwargs.get("num_init_tasks", 1)

    def compute_model_size(self, state_dict):
        return sum(p.numel() for p in state_dict.values())

    def communicate(self, task_id, communication_round):
        if task_id < self.num_init_tasks - 1:
            return
        model = self.net.state_dict()
        # send model to neighbors
        for neighbor in self.neighbors:
            neighbor.receive(self.node_id, deepcopy(model), "model")
            self.bytes_sent[(task_id, communication_round,
                             neighbor.get_node_id())] = self.compute_model_size(model)

    def receive(self, node_id, model, msg_type):
        # get model from neighbors
        # average all the models together!
        self.models[node_id] = model

    def aggregate_models(self):
        # get model from neighbors
        # average all the models together!
        print(self.models.keys())
        # average all the models together!
        for model in self.models.values():
            for name, param in model.items():
                self.net.state_dict()[name].data += param.data
        # normalize
        for name, param in self.net.state_dict().items():
            # +1 because it includes the current model
            param.data /= len(self.models) + 1

    def retrain(self, num_epochs, task_id, testloaders, save_freq=1, eval_bool=True):
        """
        Retrain on local data after aggregation. Only tested for monolithic models.
        """
        mega_dataset = ConcatDataset(
            [loader.dataset for loader in self.agent.memory_loaders.values()])
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=self.agent.memory_loaders[0].batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True
                                                  )
        self.agent._train(mega_loader, num_epochs, str(task_id) + '_retrain',
                          testloaders, save_freq, eval_bool)

    def process_communicate(self, task_id, communication_round):
        if task_id < self.num_init_tasks - 1:
            return
        self.aggregate_models()
        # train on some local tasks some more...
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=128,
                                                         shuffle=False,
                                                         num_workers=0,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}

        # NOTE: all the saving here might be a bit problematic!
        self.retrain(
            self.sharing_strategy.retrain.num_epochs, task_id, testloaders, save_freq=1, eval_bool=True)

        self.save_data(self.sharing_strategy.retrain.num_epochs + 1, task_id,
                       testloaders, final_save=True)  # final eval


@ray.remote
class ParallelModelSyncAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs)
        self.models = {}

    def communicate(self, task_id, communication_round):
        # put model on object store
        model = ray.put(self.net.state_dict())
        # send model to neighbors
        for neighbor in self.neighbors:
            neighbor.receive.remote(self.node_id, model, "model")

    def receive(self, node_id, model, msg_type):
        # get model from neighbors
        # average all the models together!
        self.models[node_id] = ray.get(model)

    def process_communicate(self, task_id):
        # get model from neighbors
        # average all the models together!
        print(self.models)
