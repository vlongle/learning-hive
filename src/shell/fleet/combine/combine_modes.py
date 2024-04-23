'''
File: /combine_modes.py
Project: combine
Created Date: Tuesday April 23rd 2024
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2024 Long Le
'''
from shell.fleet.fleet import Agent
from omegaconf import OmegaConf
from shell.fleet.utils.fleet_utils import AGENT_CLS
from shell.learners.base_learning_classes import CompositionalDynamicLearner
import ray


class CombineModesAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.communicator = self.spawn_communicator(
            self.sharing_strategy.communicator)

        self.is_modular = isinstance(self.agent, CompositionalDynamicLearner)
        self.algo = "modular" if self.is_modular else "monolithic"
        # is parallel if self is a ray actor
        self.parallel = isinstance(self, ray.actor.ActorHandle)

    def spawn_communicator(self, communicator):
        communicator = communicator.split(",")
        communicators = []
        for comm in communicator:
            # load the hydra config in epxeriments/conf/sharing_strategy/{comm}.yaml
            config_path = f"{comm}.yaml"
            config = OmegaConf.load(config_path)
            communicators.append(
                AGENT_CLS[config.name][self.algo][self.parallel])

        return communicators

    def prepare_communicate(self, task_id, end_epoch, comm_freq,
                            num_epochs,
                            communication_round, final, strategy):
        self.communicator[strategy].prepare_communicate(
            task_id, end_epoch, comm_freq, num_epochs, communication_round, final)

        self.current_strat = strategy

    def process_communicate(self, task_id, communication_round, final, strategy):
        self.communicator[strategy].prepare_communicate(
            task_id, communication_round, final)
        self.current_strat = strategy

    def communicate(
            self, task_id, communication_round, final, strategy):
        self.communicator[strategy].prepare_communicate(
            task_id, communication_round, final)
        self.current_strat = strategy

    def receive(self, node_id, data, msg_type):
        self.communicator[self.current_strat].receive(node_id, data, msg_type)


@ray.remote
class ParallelCombineModesAgent(CombineModesAgent):
    pass
