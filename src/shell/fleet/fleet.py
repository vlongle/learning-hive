'''
File: /fleet.py
Project: fleet
Created Date: Thursday March 9th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import torch
import ray
import networkx as nx
from typing import Iterable
import os
import logging
from shell.utils.utils import seed_everything, create_dir_if_not_exist

SEED_SCALE = 1000


class Agent:
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):

        self.save_dir = os.path.join(
            agent_kwargs["save_dir"], f"agent_{str(node_id)}")
        create_dir_if_not_exist(self.save_dir)

        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler(os.path.join(self.save_dir, "log.txt"))])
        self.seed = seed + SEED_SCALE * node_id
        seed_everything(self.seed)
        logging.info(
            f"Agent: node_id: {node_id}, seed: {self.seed}")
        self.node_id = node_id
        self.dataset = dataset
        self.batch_size = agent_kwargs.get("batch_size", 64)
        agent_kwargs.pop("batch_size", None)
        self.net = NetCls(**net_kwargs)
        agent_kwargs["save_dir"] = self.save_dir
        self.agent = AgentCls(self.net, **agent_kwargs)
        self.train_kwargs = train_kwargs

        self.sharing_strategy = sharing_strategy

    def get_node_id(self):
        return self.node_id

    def add_neighbors(self, neighbors: Iterable[ray.actor.ActorHandle]):
        self.neighbors = neighbors

    def train(self, task_id):
        trainloader = (
            torch.utils.data.DataLoader(self.dataset.trainset[task_id],
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        ))
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=128,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
        valloader = torch.utils.data.DataLoader(self.dataset.valset[task_id],
                                                batch_size=128,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                )
        self.agent.train(trainloader, task_id, testloaders=testloaders,
                         valloader=valloader, **self.train_kwargs)

    def communicate(self, task_id, communication_round):
        """
        Sending communication to neighbors
        """
        pass

    def process_communicate(self, task_id, communication_round):
        """
        Processing communication from neighbors
        after a round of communication
        """
        pass

    def receive(self, *args, **kwargs):
        """
        What to do when receiving data from neighbors
        """
        pass


@ray.remote
class ParallelAgent(Agent):
    pass


class Fleet:
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        self.graph = graph
        self.sharing_strategy = sharing_strategy
        self.num_coms_per_round = self.sharing_strategy.num_coms_per_round

        self.create_agents(seed, datasets, AgentCls, NetCls, LearnerCls,
                           net_kwargs, agent_kwargs, train_kwargs)
        self.add_neighbors()
        logging.info("Fleet initialized")

    def create_agents(self, seed, datasets, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        self.agents = [
            AgentCls(node_id, seed, datasets[node_id], NetCls,
                     LearnerCls, net_kwargs, agent_kwargs, train_kwargs, self.sharing_strategy)
            for node_id in self.graph.nodes
        ]
        logging.info(f"Created fleet with {len(self.agents)} agents")

    def add_neighbors(self):
        logging.info("Adding neighbors...")
        # adding neighbors
        for agent in self.agents:
            agent_id = agent.get_node_id()
            neighbors = [self.agents[neighbor_id]
                         for neighbor_id in self.graph.neighbors(agent_id)]
            agent.add_neighbors(neighbors)

    def train(self, task_id):
        for agent in self.agents:
            agent.train(task_id)

    def communicate(self, task_id):
        for communication_round in range(self.num_coms_per_round):
            for agent in self.agents:
                agent.communicate(task_id, communication_round)
            for agent in self.agents:
                agent.process_communicate(task_id, communication_round)


class ParallelFleet:
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        self.graph = graph
        num_agents = len(graph.nodes)
        num_total_gpus = torch.cuda.device_count()
        self.num_gpus_per_agent = num_total_gpus / num_agents
        logging.info(f"No. gpus per agent: {self.num_gpus_per_agent}")
        self.sharing_strategy = sharing_strategy
        self.num_coms_per_round = self.sharing_strategy.num_coms_per_round

        self.create_agents(seed, datasets, AgentCls, NetCls, LearnerCls,
                           net_kwargs, agent_kwargs, train_kwargs)
        self.add_neighbors()

        logging.info("Fleet initialized")

    def create_agents(self, seed, datasets, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        self.agents = [
            AgentCls.options(num_gpus=self.num_gpus_per_agent).remote(node_id, seed, datasets[node_id], NetCls,
                                                                      LearnerCls, net_kwargs, agent_kwargs, train_kwargs, self.sharing_strategy)
            for node_id in self.graph.nodes
        ]
        logging.info(f"Created fleet with {len(self.agents)} agents")

    def add_neighbors(self):
        logging.info("Adding neighbors...")
        # adding neighbors
        for agent in self.agents:
            agent_id = ray.get(agent.get_node_id.remote())
            neighbors = [self.agents[neighbor_id]
                         for neighbor_id in self.graph.neighbors(agent_id)]
            agent.add_neighbors.remote(neighbors)

    def train(self, task_id):
        # parallelize training
        ray.get([agent.train.remote(task_id) for agent in self.agents])

    def communicate(self, task_id):
        for communication_round in range(self.num_coms_per_round):
            ray.get([agent.communicate.remote(task_id, communication_round)
                    for agent in self.agents])

            ray.get([agent.process_communicate.remote(task_id, communication_round)
                    for agent in self.agents])
