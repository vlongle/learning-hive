'''
File: /fedcurv.py
Project: grad
Created Date: Friday April 19th 2024
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2024 Long Le
'''

from shell.fleet.grad.monograd import *
from shell.fleet.grad.fedcurv_utils import EWC
import copy
import torch


class FedCurvAgent(ModelSyncAgent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs,
                 sharing_strategy, agent=None, net=None):
        if "fl_strategy" not in agent_kwargs:
            agent_kwargs["fl_strategy"] = "fedcurv"
        self.incoming_fishers = {}
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy, agent=agent, net=net)

    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True):
        self.agent.mu = self.sharing_strategy.mu
        return super().train(task_id, start_epoch, communication_frequency, final)

    def prepare_communicate(self, task_id, end_epoch, comm_freq, num_epochs,
                            communication_round, final=False):
        super().prepare_communicate(task_id, end_epoch, comm_freq, num_epochs,
                                    communication_round, final)

        tmp_dataset = copy.deepcopy(self.dataset.trainset[task_id])
        tmp_dataset.tensors = tmp_dataset.tensors + \
            (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        mega_dataset = ConcatDataset(
            [get_custom_tensordataset(replay.get_tensors(), name=self.dataset.name,
                                      use_contrastive=self.agent.use_contrastive) for replay in self.agent.replay_buffers.values()] + [tmp_dataset]
            + [get_custom_tensordataset(replay.get_tensors(), name=self.dataset.name,
                                        use_contrastive=self.agent.use_contrastive) for replay in self.agent.shared_replay_buffers.values() if len(replay) > 0]
        )
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  pin_memory=True
                                                  )

        self.fisher = EWC(self.agent, mega_loader).fisher

    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            neighbor.receive(self.node_id, deepcopy(self.model), "model")
            neighbor.receive(self.node_id, deepcopy(self.fisher), "fisher")

    def receive(self, node_id, model, msg_type):
        if msg_type == "model":
            self.incoming_models[node_id] = model
        elif msg_type == "fisher":
            self.incoming_fishers[node_id] = model
        else:
            raise ValueError(f"Invalid message type: {msg_type}")

    def process_communicate(self, task_id, communication_round, final=False):
        self.agent.incoming_models = self.incoming_models
        self.agent.mu = self.sharing_strategy.mu
        self.agent.fisher = self.incoming_fishers
        return super().process_communicate(task_id, communication_round, final)


@ray.remote
class ParallelFedCurvAgent(FedCurvAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.fisher), "fisher"))


class FedCurvModAgent(FedCurvAgent):
    def prepare_model(self):
        num_init_components = self.net.depth
        num_components = len(self.net.components)
        for i in range(num_init_components, num_components):
            self.excluded_params.add("components.{}".format(i))
        return super().prepare_model()


@ray.remote
class ParallelFedCurvModAgent(FedCurvModAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.fisher), "fisher"))
