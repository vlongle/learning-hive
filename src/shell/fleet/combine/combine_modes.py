'''
File: /combine_modes.py
Project: combine
Created Date: Tuesday April 23rd 2024
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2024 Long Le
'''
from shell.fleet.fleet import Agent
from shell.learners.base_learning_classes import CompositionalDynamicLearner
import ray

from copy import deepcopy
from shell.fleet.combine.combine_modes_utils import *
import logging


# NOTE: need to double check that communicator is able to modify the underlying kwargs for prepare_train

class CombineModesAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy,
                 ):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.is_modular = isinstance(self.agent, CompositionalDynamicLearner)
        self.algo = "modular" if self.is_modular else "monolithic"
        # is parallel if self is a ray actor
        self.NetCls = NetCls
        self.AgentCls = AgentCls
        self.net_kwargs = net_kwargs
        self.agent_kwargs = agent_kwargs
        self.train_kwargs = train_kwargs
        self.sharing_strategy = sharing_strategy

        # self.communicator = self.spawn_communicator(
        #     self.sharing_strategy.communicator)

    def spawn_communicator(self, communicator=None):
        if communicator is None:
            communicator = self.sharing_strategy.communicator
        communicator = communicator.split(",")
        self.communicator = {}
        for comm in communicator:
            config = load_comm_config(comm)
            a_kw = deepcopy(self.agent_kwargs)
            a_kw['save_dir'] = self.root_save_dir
            args = (self.node_id, self.seed, self.dataset, self.NetCls, self.AgentCls,
                    deepcopy(self.net_kwargs), a_kw, self.train_kwargs, config, self.agent)

            comm_cls = AGENT_CLS[config.name][self.algo][False]

            comm_agent = comm_cls(*args)
            comm_agent.add_neighbors(self.neighbors)

            self.communicator[comm] = comm_agent

    def prepare_communicate(self, task_id, end_epoch, comm_freq,
                            num_epochs,
                            communication_round, final, strategy):
        self.communicator[strategy].prepare_communicate(
            task_id, end_epoch, comm_freq, num_epochs, communication_round, final)

        self.current_strat = strategy

    def process_communicate(self, task_id, communication_round, final, strategy):
        self.communicator[strategy].process_communicate(
            task_id, communication_round, final)
        self.current_strat = strategy

    def communicate(
            self, task_id, communication_round, final, strategy):
        self.communicator[strategy].communicate(
            task_id, communication_round, final)
        self.current_strat = strategy

    def receive(self, node_id, data, msg_type):
        self.communicator[self.current_strat].receive(node_id, data, msg_type)


@ ray.remote
class ParallelCombineModesAgent(CombineModesAgent):
    pass
    # def communicate(self, task_id, communication_round, final=False, strategy=None):
    #     communicator = self.communicator[strategy]
    #     for neighbor in self.neighbors.values():
    #         ray.get(neighbor.receive.remote(
    #             self.node_id, deepcopy(self.model), "model"))
