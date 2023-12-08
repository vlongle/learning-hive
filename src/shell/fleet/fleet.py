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
from typing import Any, Iterable
import os
import logging
from shell.utils.utils import seed_everything, create_dir_if_not_exist
from shell.utils.experiment_utils import eval_net
from copy import deepcopy
import math
import pandas as pd

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
        self.num_coms = {}
        self.node_id = node_id
        self.dataset = dataset
        self.batch_size = agent_kwargs.get("batch_size", 64)
        agent_kwargs.pop("batch_size", None)
        self.net = NetCls(**net_kwargs)
        agent_kwargs["save_dir"] = self.save_dir
        self.agent = AgentCls(self.net, **agent_kwargs)
        self.train_kwargs = train_kwargs

        self.sharing_strategy = sharing_strategy

    def set_num_coms(self, task_id, num_coms):
        self.num_coms[task_id] = num_coms

    def get_node_id(self):
        return self.node_id

    def set_fl_strategy(self, fl_strategy):
        self.agent.fl_strategy = fl_strategy

    def add_neighbors(self, neighbors: Iterable[ray.actor.ActorHandle]):
        self.neighbors = neighbors

    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True):
        if task_id >= self.net.num_tasks:
            return
        trainloader = (
            torch.utils.data.DataLoader(self.dataset.trainset[task_id],
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        ))
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=256,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
        valloader = torch.utils.data.DataLoader(self.dataset.valset[task_id],
                                                batch_size=256,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                )
        train_kwargs = self.train_kwargs.copy()

        # use `init_num_epochs` and `init_component_update_freq` for the first few tasks
        num_epochs, component_update_freq = None, None
        if "init_num_epochs" in train_kwargs:
            num_epochs = train_kwargs.pop(
                "init_num_epochs")
        if "init_component_update_freq" in train_kwargs:
            component_update_freq = train_kwargs.pop(
                "init_component_update_freq")
        if task_id < self.agent.net.num_init_tasks:
            if num_epochs is not None:
                train_kwargs["num_epochs"] = num_epochs
            if component_update_freq is not None:
                train_kwargs["component_update_freq"] = component_update_freq

        if communication_frequency is None:
            # communication_frequency = train_kwargs['num_epochs'] - start_epoch
            communication_frequency = train_kwargs['num_epochs'] - start_epoch

        end_epoch = min(start_epoch + communication_frequency,
                        train_kwargs['num_epochs'])
        adjusted_num_epochs = end_epoch - start_epoch
        train_kwargs["num_epochs"] = adjusted_num_epochs
        train_kwargs["final"] = final

        self.agent.train(trainloader, task_id, testloaders=testloaders,
                         valloader=valloader, start_epoch=start_epoch, **train_kwargs)

    def eval_test(self, task_id):
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=128,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
        return eval_net(self.net, testloaders)

    def eval_val(self, task_id):
        valloaders = {task: torch.utils.data.DataLoader(valset,
                                                        batch_size=128,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        ) for task, valset in enumerate(self.dataset.valset[:(task_id+1)])}
        return eval_net(self.net, valloaders)

    def communicate(self, task_id, communication_round, final=False):
        """
        Sending communication to neighbors
        """
        pass

    def prepare_communicate(self, task_id, communication_round, final=False):
        """
        Preparing communication to neighbors
        """
        pass

    def process_communicate(self, task_id, communication_round, final=False):
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

    def get_net(self):
        """
        Used for reading only, because technically actor
        might live on a different name space (node/machine).
        """
        return self.net

    def load_and_freeze_random_linear_projection(self, state_dict):
        self.net.load_and_freeze_random_linear_projection(state_dict)

    def get_model(self):
        return self.net.state_dict()

    def get_num_components(self):
        return len(self.net.components)

    def replace_dataset(self, dataset, task):
        self.dataset.trainset[task] = dataset.trainset[task]
        self.dataset.testset[task] = dataset.testset[task]
        self.dataset.valset[task] = dataset.valset[task]
        if not self.dataset.class_sequence.flags.writeable:
            self.dataset.class_sequence = self.dataset.class_sequence.copy()
        self.dataset.class_sequence[:task * (self.dataset.num_classes_per_task)
                                    ] = dataset.class_sequence[:task * (self.dataset.num_classes_per_task)]

    def replace_model(self, new_model, strict=True):
        # print("replacing model with strict:", strict)
        self.net.load_state_dict(new_model, strict=strict)
        self.net.to(self.net.device)

    def replace_record(self, record):
        self.agent.record.df = record.df.copy()
        self.agent.record.save()

    def replace_replay(self, trainloader, task_id, increment_T=True):
        self.agent.update_multitask_cost(trainloader, task_id)
        self.agent.observed_tasks.add(task_id)
        if increment_T:
            self.agent.T += 1

    def get_record(self):
        return self.agent.record

    def get_num_components(self, agent_path, task_id):
        # agent_path = {something}/agent_{node_id}
        add_modules_record = os.path.join(
            agent_path, "add_modules_record.csv")
        df = pd.read_csv(add_modules_record)
        return df[df["task_id"] == task_id]["num_components"].sum()

    def load_model_from_ckpoint(self, task_path=None, task_id=None):
        # path = {something}/agent_{node_id}/task_{task_id}
        if task_path is None:
            if task_id is None:
                task_id = max([int(folder.split("_")[1]) for folder in os.listdir(
                    self.save_dir) if folder.startswith("task")])
            task_path = os.path.join(
                self.save_dir, 'task_{}'.format(task_id))
        else:
            # get task_id from task_path
            task_id = int(task_path.split("_")[-1])

        print("Loading model from ckpoint", task_path)
        agent_path = os.path.dirname(task_path)
        get_num_components = self.get_num_components(
            agent_path, task_id)
        num_added_components = get_num_components - \
            len(self.net.components)
        for _ in range(num_added_components):
            self.net.add_tmp_modules(task_id=len(
                self.net.components), num_modules=1)

        self.net.load_state_dict(torch.load(os.path.join(
            task_path, "checkpoint.pt"))['model_state_dict'])

    def update_replay_buffer(self, task_id):
        self.agent.replay_buffers = {}
        self.agent.aug_replay_buffers = {}
        for task in range(task_id+1):
            trainloader = torch.utils.data.DataLoader(self.dataset.trainset[task],
                                                      batch_size=128,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      )
            self.agent.update_multitask_cost(trainloader, task)
            # self.agent.T += 1
        self.agent.T = len(self.agent.replay_buffers)


@ray.remote
class ParallelAgent(Agent):
    pass


class Fleet:
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs,
                 agent_kwargs, train_kwargs):
        self.graph = graph
        self.sharing_strategy = sharing_strategy
        self.num_coms_per_round = self.sharing_strategy.num_coms_per_round

        self.create_agents(seed, datasets, AgentCls, NetCls, LearnerCls,
                           net_kwargs, agent_kwargs, train_kwargs)
        self.add_neighbors()
        self.num_epochs = train_kwargs["num_epochs"]
        self.init_num_epochs = train_kwargs.get(
            "init_num_epochs", self.num_epochs)
        self.comm_freq = sharing_strategy.get("comm_freq", None)
        self.num_init_tasks = net_kwargs["num_init_tasks"]

        logging.info("Fleet initialized")

    def eval_test(self, task_id=None):
        if task_id is None:
            task_id = self.agents[0].agent.T - 1
        return [agent.eval_test(task_id) for agent in self.agents]

    def eval_val(self, task_id=None):
        if task_id is None:
            task_id = self.agents[0].agent.T - 1
        return [agent.eval_val(task_id) for agent in self.agents]

    def load_model_from_ckpoint(self, paths=None, task_ids=None):
        if paths is None:
            paths = [None] * len(self.agents)
        if task_ids is None:
            task_ids = [None] * len(self.agents)
        if isinstance(task_ids, int):
            task_ids = [task_ids] * len(self.agents)
        for agent, path, task_id in zip(self.agents, paths, task_ids):
            agent.load_model_from_ckpoint(task_path=path, task_id=task_id)

    def update_replay_buffers(self, task_id):
        for agent in self.agents:
            agent.update_replay_buffer(task_id)

    def create_agents(self, seed, datasets, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs,
                      train_kwargs, uniformized=False):
        self.agents = [
            AgentCls(node_id, seed, datasets[node_id], NetCls,
                     LearnerCls, deepcopy(net_kwargs), deepcopy(agent_kwargs),
                     deepcopy(train_kwargs), deepcopy(self.sharing_strategy))
            for node_id in self.graph.nodes
        ]
        # make sure that all agents share the same (random) preprocessing parameters in MNIST variants
        # check that self.agents[0].net has "random_linear_projection" layer, if yes, then share it
        # among all other agents
        if hasattr(self.agents[0].net, "random_linear_projection"):
            for agent in self.agents[1:]:
                agent.load_and_freeze_random_linear_projection(
                    self.agents[0].net.random_linear_projection.state_dict())
        logging.info(f"Created fleet with {len(self.agents)} agents")

    def add_neighbors(self):
        logging.info("Adding neighbors...")
        # adding neighbors
        for agent in self.agents:
            agent_id = agent.get_node_id()
            neighbors = [self.agents[neighbor_id]
                         for neighbor_id in self.graph.neighbors(agent_id)]
            agent.add_neighbors(neighbors)

    def train_and_comm(self, task_id):

        if task_id < self.num_init_tasks:
            # init task
            num_epochs = self.init_num_epochs
        else:
            num_epochs = self.num_epochs
        comm_freq = self.comm_freq if self.comm_freq is not None else num_epochs + 1

        # Number of times the loop will iterate
        num_coms = math.ceil(num_epochs / comm_freq)

        for start_epoch in range(0, num_epochs, comm_freq):
            for agent in self.agents:
                # only remove modules for the last epoch
                final = start_epoch + comm_freq >= num_epochs
                agent.set_num_coms(task_id, num_coms)
                agent.train(task_id, start_epoch, comm_freq, final=final)

            end_epoch = min(start_epoch + comm_freq, num_epochs)
            if comm_freq <= num_epochs and (end_epoch % comm_freq == 0):
                print('COMMUNICATING')
                self.communicate(task_id if not final else task_id + 1,
                                 start_com_round=(
                                     start_epoch // comm_freq) * self.num_coms_per_round,
                                 final=final)

    def communicate(self, task_id, start_com_round=0, final=False):
        for communication_round in range(start_com_round, self.num_coms_per_round + start_com_round):
            for agent in self.agents:
                agent.prepare_communicate(task_id, communication_round, final)
            for agent in self.agents:
                agent.communicate(task_id, communication_round, final)
            for agent in self.agents:
                agent.process_communicate(task_id, communication_round, final)


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
        self.num_epochs = train_kwargs["num_epochs"]
        self.init_num_epochs = train_kwargs.get(
            "init_num_epochs", self.num_epochs)
        self.comm_freq = sharing_strategy.get("comm_freq", None)
        self.num_init_tasks = net_kwargs["num_init_tasks"]

        logging.info("Fleet initialized")

    def load_model_from_ckpoint(self, paths=None, task_ids=None):
        if paths is None:
            paths = [None] * len(self.agents)
        if task_ids is None:
            task_ids = [None] * len(self.agents)
        for agent, path, task_id in zip(self.agents, paths, task_ids):
            ray.get(agent.load_model_from_ckpoint.remote(
                task_path=path, task_id=task_id))

    def create_agents(self, seed, datasets, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        self.agents = [
            AgentCls.options(num_gpus=self.num_gpus_per_agent).remote(node_id, seed, datasets[node_id], NetCls,
                                                                      LearnerCls,
                                                                      deepcopy(net_kwargs), deepcopy(
                                                                          agent_kwargs),
                                                                      deepcopy(train_kwargs), deepcopy(self.sharing_strategy))
            for node_id in self.graph.nodes
        ]

        # make sure that all agents share the same (random) preprocessing parameters in MNIST variants
        # check that self.agents[0].net has "random_linear_projection" layer, if yes, then share it
        # among all other agents
        # use get_net to get the net from the remote agent
        net0 = ray.get(self.agents[0].get_net.remote())
        if hasattr(net0, "random_linear_projection"):
            rand_lp = net0.random_linear_projection.state_dict()
            for agent in self.agents[1:]:
                agent.load_and_freeze_random_linear_projection.remote(rand_lp)

        logging.info(f"Created fleet with {len(self.agents)} agents")

    def add_neighbors(self):
        logging.info("Adding neighbors...")
        # adding neighbors
        for agent in self.agents:
            agent_id = ray.get(agent.get_node_id.remote())
            neighbors = [self.agents[neighbor_id]
                         for neighbor_id in self.graph.neighbors(agent_id)]
            agent.add_neighbors.remote(neighbors)

    def train_and_comm(self, task_id):
        if task_id < self.num_init_tasks:
            # init task
            num_epochs = self.init_num_epochs
        else:
            num_epochs = self.num_epochs
        comm_freq = self.comm_freq if self.comm_freq is not None else num_epochs + 1
        # Number of times the loop will iterate
        num_coms = math.ceil(num_epochs / comm_freq)

        for start_epoch in range(0, num_epochs, comm_freq):
            final = start_epoch + comm_freq >= num_epochs
            ray.get([agent.set_num_coms.remote(
                task_id, num_coms) for agent in self.agents])
            ray.get([agent.train.remote(task_id, start_epoch, comm_freq,
                                        final=final) for agent in self.agents])

            end_epoch = min(start_epoch + comm_freq, num_epochs)
            if comm_freq <= num_epochs and (end_epoch % comm_freq == 0):
                self.communicate(task_id if not final else task_id + 1,
                                 start_com_round=(
                                     start_epoch // comm_freq) * self.num_coms_per_round,
                                 final=final)

    def communicate(self, task_id, start_com_round=0, final=False):
        for communication_round in range(start_com_round, self.num_coms_per_round + start_com_round):
            # parallelize preprocessing to prepare neccessary data
            # before the communication round.
            ray.get([agent.prepare_communicate.remote(task_id, communication_round, final)
                    for agent in self.agents])
            # NOTE: HACK: communicate in done in sequence to avoid dysnc issues,
            # if the sender sends something to the receiver but the receiver is not paying
            # attention, then the message is lost. Truly decen agent would have to have a
            # while True loop and implement some more complicated logic.
            for agent in self.agents:
                # sequential, because ray.get is blocking to get the result from one
                # agent before moving to the next.
                ray.get(agent.communicate.remote(
                    task_id, communication_round, final))

            ray.get([agent.process_communicate.remote(task_id, communication_round, final)
                    for agent in self.agents])
