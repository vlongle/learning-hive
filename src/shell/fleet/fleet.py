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
from shell.fleet.data.data_utilize import *
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from shell.datasets.datasets import get_custom_tensordataset

SEED_SCALE = 1000


class Agent:
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):

        self.seed = seed + SEED_SCALE * node_id
        seed_everything(self.seed)

        self.save_dir = os.path.join(
            agent_kwargs["save_dir"], f"agent_{str(node_id)}")
        create_dir_if_not_exist(self.save_dir)

        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler(os.path.join(self.save_dir, "log.txt"))])
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

    # def get_ood_data(self, task_id, mode='replay'):

    #     if mode == 'replay':
    #         # Gather data from replay buffers of all tasks except the current task
    #         replay_buffers = {t: self.agent.replay_buffers[t] for t in range(self.agent.T)
    #                           if t != task_id}
    #         if len(replay_buffers) == 0:
    #             return None, None, None, None

    #         X_past = torch.cat([rb.tensors[0]
    #                             for t, rb in replay_buffers.items()], dim=0)
    #         y_past = torch.cat([torch.from_numpy(get_global_label(rb.tensors[1],
    #                                                               t, self.dataset.class_sequence,
    #                                                               self.dataset.num_classes_per_task))
    #                             for t, rb in replay_buffers.items()], dim=0)
    #     elif mode == 'training':
    #         # Gather data from training sets of all tasks except the current task
    #         X_past = torch.cat([self.dataset.trainset[t].tensors[0]
    #                             for t in range(self.agent.T) if t != task_id], dim=0)
    #         y_past = torch.cat([torch.from_numpy(get_global_label(self.dataset.trainset[t].tensors[1],
    #                                                               t, self.dataset.class_sequence,
    #                                                               self.dataset.num_classes_per_task))
    #                             for t in range(self.agent.T) if t != task_id], dim=0)
    #     mask = self.get_ood_data_helper(task_id, y_past)
    #     X_ood_filtered = X_past[mask]
    #     y_ood_filtered = y_past[mask]

    #     X_iid_filtered = X_past[~mask]
    #     y_iid_filtered = y_past[~mask]

    #     return X_ood_filtered, y_ood_filtered, X_iid_filtered, y_iid_filtered

    def get_shared_replay_buffers(self):
        return self.agent.shared_replay_buffers

    def get_replay_buffers(self):
        return self.agent.replay_buffers

    def get_T(self):
        return self.agent.T

    def get_task_class(self, task_id):
        # Get the class labels for the current task
        task_classes = list(self.dataset.class_sequence[task_id * self.dataset.num_classes_per_task:
                                                        (task_id + 1) * self.dataset.num_classes_per_task])
        return task_classes

    def get_all_classes(self, task_id):
        return set([c for t in range(task_id+1) for c in self.get_task_class(t)])

    def get_ood_data_helper(self, task_id, candidate_ys):
        task_classes = self.get_task_class(task_id)
        # Convert task_classes to a tensor for efficient comparison
        task_classes_tensor = torch.tensor(task_classes)

        # Find indices of samples in y_ood that do not belong to the current task's classes
        mask = ~candidate_ys.unsqueeze(1).eq(task_classes_tensor).any(1)
        return mask

    def set_num_coms(self, task_id, num_coms):
        self.num_coms[task_id] = num_coms

    def get_node_id(self):
        return self.node_id

    def set_fl_strategy(self, fl_strategy):
        self.agent.fl_strategy = fl_strategy

    def get_fl_strategy(self):
        return self.agent.fl_strategy

    def add_neighbors(self, neighbors: Iterable[ray.actor.ActorHandle]):
        self.neighbors = neighbors

    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True, **kwargs):
        # if start_epoch == 0:
        #     for t in range(task_id+1):
        #         self.agent.ood_data[t] = self.get_ood_data(t)

        if task_id >= self.net.num_tasks:
            return

        # dataset = deepcopy(self.dataset.trainset[task_id])

        # self.agent.make_shared_memory_loaders(
        #     batch_size=self.batch_size)

        # if task_id in self.agent.shared_memory_loaders:
        #     loader = self.agent.shared_memory_loaders[task_id]
        #     shared_tensors = loader.dataset.get_tensors()  # X, y, t
        #     # throw away the task id
        #     shared_tensors = shared_tensors[:2]
        #     dataset = CustomConcatDataset(
        #         dataset.tensors, shared_tensors)
        #     dataset = get_custom_tensordataset(dataset.tensors, name=self.dataset.name,
        #                                        use_contrastive=self.agent.use_contrastive)

        # trainloader = (
        #     torch.utils.data.DataLoader(dataset,
        #                                 batch_size=self.batch_size,
        #                                 shuffle=True,
        #                                 num_workers=4,
        #                                 pin_memory=True,
        #                                 ))

        trainloader = (
            torch.utils.data.DataLoader(self.dataset.trainset[task_id],
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        # shuffle=False,
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

        train_kwargs.update(kwargs)

        # print('train task', task_id, 'rand torch seed', int(torch.empty(
        #     (), dtype=torch.int64).random_().item()))

        return self.agent.train(trainloader, task_id, testloaders=testloaders,
                                valloader=valloader, start_epoch=start_epoch, **train_kwargs)

    def eval_test(self, task_id, include_avg=False):
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=128,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
        return eval_net(self.net, testloaders, include_avg=include_avg)

    # def eval_val(self, task_id):
    #     valloaders = {task: torch.utils.data.DataLoader(valset,
    #                                                     batch_size=128,
    #                                                     shuffle=False,
    #                                                     num_workers=4,
    #                                                     pin_memory=True,
    #                                                     ) for task, valset in enumerate(self.dataset.valset[:(task_id+1)])}
    #     return eval_net(self.net, valloaders)

    def eval_val(self, tasks):
        valloaders = {task: torch.utils.data.DataLoader(self.dataset.valset[task],
                                                        batch_size=128,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        ) for task in tasks}
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

    def replace_dataset(self, dataset, task):
        self.dataset.trainset[task] = dataset.trainset[task]
        self.dataset.testset[task] = dataset.testset[task]
        self.dataset.valset[task] = dataset.valset[task]
        if not self.dataset.class_sequence.flags.writeable:
            self.dataset.class_sequence = self.dataset.class_sequence.copy()

        self.dataset.class_sequence[task *
                                    self.dataset.num_classes_per_task: (task + 1) * self.dataset.num_classes_per_task] = \
            dataset.class_sequence[task * dataset.num_classes_per_task: (
                task + 1) * dataset.num_classes_per_task]

    def replace_model(self, new_model, strict=True):
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
        if "monolithic" in agent_path:
            return len(self.net.components)
        # agent_path = {something}/agent_{node_id}
        add_modules_record = os.path.join(
            agent_path, "add_modules_record.csv")
        df = pd.read_csv(add_modules_record)
        return df[df["task_id"] == task_id]["num_components"].sum()

    def load_records(self):
        perf_record = os.path.join(self.save_dir, "record.csv")
        self.agent.record.df = pd.read_csv(
            perf_record
        )
        add_modules_record = os.path.join(
            self.agent.save_dir, "add_modules_record.csv")
        if os.path.exists(add_modules_record):
            try:
                self.agent.dynamic_record.df = pd.read_csv(add_modules_record)
            except pd.errors.EmptyDataError:
                pass

        sharing_data_record = os.path.join(
            self.agent.save_dir, "sharing_data_record.csv")
        if os.path.exists(sharing_data_record):
            try:
                self.agent.sharing_data_record.df = pd.read_csv(
                    sharing_data_record)
            except pd.errors.EmptyDataError:
                pass

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

        logging.debug("Loading model from ckpoint {task_path}")
        agent_path = os.path.dirname(task_path)
        get_num_components = self.get_num_components(
            agent_path, task_id)
        num_added_components = get_num_components - \
            len(self.net.components)
        # print('get_num_components', get_num_components, 'num_added_components',
        #       num_added_components, 'len(self.net.components)', len(self.net.components))
        # print(self.save_dir)
        for _ in range(num_added_components):
            self.net.add_tmp_modules(task_id=len(
                self.net.components), num_modules=1)

        self.net.load_state_dict(torch.load(os.path.join(
            task_path, "checkpoint.pt"))['model_state_dict'])

    def update_replay_buffer(self, task_id):
        self.agent.replay_buffers = {}
        for task in range(task_id+1):
            trainloader = torch.utils.data.DataLoader(self.dataset.trainset[task],
                                                      batch_size=128,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      pin_memory=True,
                                                      )
            self.agent.update_multitask_cost(trainloader, task)
            # self.agent.T += 1
        self.agent.T = len(self.agent.replay_buffers)

    def change_save_dir(self, save_dir):
        self.agent.change_save_dir(save_dir)

    def get_save_dir(self):
        return self.agent.save_dir

    def get_dataset_name(self):
        return self.dataset.name


@ray.remote
class ParallelAgent(Agent):
    pass


class Fleet:
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs,
                 agent_kwargs, train_kwargs):
        self.graph = graph
        self.sharing_strategy = sharing_strategy
        self.num_coms_per_round = self.sharing_strategy.num_coms_per_round

        self.remove_ood_neighbors = getattr(
            self.sharing_strategy, 'remove_ood_neighbors', False)
        self.create_agents(seed, datasets, AgentCls, NetCls, LearnerCls,
                           net_kwargs, agent_kwargs, train_kwargs)
        self.add_neighbors()
        self.num_epochs = train_kwargs["num_epochs"]
        self.init_num_epochs = train_kwargs.get(
            "init_num_epochs", self.num_epochs)
        self.comm_freq = sharing_strategy.get("comm_freq", None)
        self.num_init_tasks = net_kwargs["num_init_tasks"]
        self.num_tasks = net_kwargs["num_tasks"]

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

    def load_records(self):
        for agent in self.agents:
            agent.load_records()

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
        for agent in self.agents:
            agent_id = agent.get_node_id()
            neighbors = {}
            for neighbor_id in self.graph.neighbors(agent_id):
                if self.remove_ood_neighbors:
                    # Only add the neighbor if the dataset names match
                    if agent.get_dataset_name() == self.agents[neighbor_id].get_dataset_name():
                        neighbors[self.agents[neighbor_id].get_node_id(
                        )] = self.agents[neighbor_id]
                else:
                    neighbors[self.agents[neighbor_id].get_node_id()
                              ] = self.agents[neighbor_id]
            agent.add_neighbors(neighbors)

    def train_and_comm(self, task_id):
        if task_id < self.num_init_tasks:
            # init task
            num_epochs = self.init_num_epochs
        else:
            num_epochs = self.num_epochs

        if isinstance(self.comm_freq, dict):
            comm_freqs = self.comm_freq
        else:
            comm_freqs = {'default': self.comm_freq} if self.comm_freq is not None else {
                'default': num_epochs + 1}

        # Create a combined list of all unique communication epochs, including the last epoch
        unique_epochs = set()
        for freq in comm_freqs.values():
            unique_epochs.update(range(freq, num_epochs + 1, freq))
        # Ensure the last epoch is always included
        unique_epochs.add(num_epochs)
        sorted_epochs = sorted(unique_epochs)

        max_comm_freq = max(comm_freqs.values())
        num_coms = math.ceil(num_epochs / max_comm_freq)

        start_epoch = 0
        for end_epoch in sorted_epochs:
            final = end_epoch == num_epochs
            print('from', start_epoch, 'to', end_epoch, 'final', final)

            if self.sharing_strategy.pre_or_post_comm == "pre":
                for strategy, freq in comm_freqs.items():
                    if end_epoch % freq == 0 and freq <= num_epochs:
                        logging.info(
                            f'>>> {strategy.upper()} COMM AT EPOCH', end_epoch)
                        self.communicate(
                            task_id, end_epoch, freq, num_epochs, strategy=strategy, final=final)

            for agent in self.agents:
                agent.set_num_coms(task_id, num_coms)
                agent.train(task_id, start_epoch, end_epoch -
                            start_epoch, final=final)

            if self.sharing_strategy.pre_or_post_comm == "post":
                for strategy, freq in comm_freqs.items():
                    if end_epoch % freq == 0 and freq <= num_epochs:
                        logging.info(
                            f'>>> {strategy.upper()} COMM AT EPOCH', end_epoch)
                        self.communicate(
                            task_id, end_epoch, freq, num_epochs, strategy=strategy, final=final)

            start_epoch = end_epoch

    def communicate(self, task_id, end_epoch, comm_freq, num_epochs, start_com_round=0, final=False, strategy=None):
        for communication_round in range(start_com_round, self.num_coms_per_round + start_com_round):
            self.communicate_round(
                task_id, end_epoch, comm_freq, num_epochs, communication_round, final=final, strategy=strategy)

    def communicate_round(self, task_id, end_epoch, comm_freq, num_epochs, communication_round, final=False, strategy=None):
        for agent in self.agents:
            agent.prepare_communicate(
                task_id, end_epoch, comm_freq, num_epochs, communication_round, final, strategy=strategy)
        for agent in self.agents:
            agent.communicate(task_id, communication_round,
                              final, strategy=strategy)
        for agent in self.agents:
            agent.process_communicate(
                task_id, communication_round, final, strategy=strategy)

    def get_save_dir(self):
        return [agent.get_save_dir() for agent in self.agents]

    def change_save_dir(self, save_dir):
        for agent in self.agents:
            node_id = agent.get_node_id()
            agent_save_dir = os.path.join(save_dir, f"agent_{node_id}")
            agent.change_save_dir(agent_save_dir)


class ParallelFleet:
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        self.graph = graph
        num_agents = len(graph.nodes)
        num_total_gpus = torch.cuda.device_count()
        self.num_gpus_per_agent = num_total_gpus / num_agents
        logging.info(f"No. gpus per agent: {self.num_gpus_per_agent}")
        self.sharing_strategy = sharing_strategy
        self.num_coms_per_round = self.sharing_strategy.num_coms_per_round

        self.remove_ood_neighbors = getattr(
            self.sharing_strategy, 'remove_ood_neighbors', False)

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
        if isinstance(task_ids, int):
            task_ids = [task_ids] * len(self.agents)
        for agent, path, task_id in zip(self.agents, paths, task_ids):
            ray.get(agent.load_model_from_ckpoint.remote(
                task_path=path, task_id=task_id))

    def update_replay_buffers(self, task_id):
        for agent in self.agents:
            ray.get(agent.update_replay_buffer.remote(task_id))

    def eval_test(self, task_id=None):
        if task_id is None:
            task_id = ray.get(self.agents[0].get_T.remote()) - 1
        return ray.get([agent.eval_test.remote(task_id) for agent in self.agents])

    def create_agents(self, seed, datasets, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs):
        print('CREATING AGENTS...')
        self.agents = [
            AgentCls.options(num_gpus=self.num_gpus_per_agent).remote(node_id, seed, datasets[node_id], NetCls,
                                                                      LearnerCls,
                                                                      deepcopy(net_kwargs), deepcopy(
                agent_kwargs),
                deepcopy(train_kwargs), deepcopy(self.sharing_strategy))
            for node_id in self.graph.nodes
        ]
        print('DONE AGENTS...')

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
        for agent in self.agents:
            agent_id = ray.get(agent.get_node_id.remote())
            neighbors = {}
            for neighbor_id in self.graph.neighbors(agent_id):
                if self.remove_ood_neighbors:
                    # Retrieve dataset names asynchronously and compare
                    if ray.get(agent.get_dataset_name.remote()) == ray.get(self.agents[neighbor_id].get_dataset_name.remote()):
                        neighbors[ray.get(
                            self.agents[neighbor_id].get_node_id.remote())] = self.agents[neighbor_id]
                else:
                    neighbors[ray.get(
                        self.agents[neighbor_id].get_node_id.remote())] = self.agents[neighbor_id]
            agent.add_neighbors.remote(neighbors)

    def train_and_comm(self, task_id):
        if task_id < self.num_init_tasks:
            # Initialization task
            num_epochs = self.init_num_epochs
        else:
            num_epochs = self.num_epochs

        # Handling different communication frequencies
        if isinstance(self.comm_freq, dict):
            comm_freqs = self.comm_freq
        else:
            comm_freqs = {'default': self.comm_freq} if self.comm_freq is not None else {
                'default': num_epochs + 1}

        # Create a combined list of all unique communication epochs
        unique_epochs = set()
        for freq in comm_freqs.values():
            unique_epochs.update(range(freq, num_epochs + 1, freq))
        # Ensure the last epoch is always included
        unique_epochs.add(num_epochs)
        sorted_epochs = sorted(unique_epochs)

        # Number of times the loop will iterate
        max_comm_freq = max(comm_freqs.values())
        num_coms = math.ceil(num_epochs / max_comm_freq)

        start_epoch = 0
        for end_epoch in sorted_epochs:
            final = end_epoch == num_epochs
            logging.info(
                f'Task {task_id} training from {start_epoch} to {end_epoch}')
            ray.get([agent.set_num_coms.remote(task_id, num_coms)
                    for agent in self.agents])
            ray.get([agent.train.remote(task_id, start_epoch, end_epoch -
                    start_epoch, final=final) for agent in self.agents])

            if self.sharing_strategy.pre_or_post_comm == "pre":
                for strategy, freq in comm_freqs.items():
                    if end_epoch % freq == 0 and freq <= num_epochs:
                        logging.info(
                            f'Task {task_id} {strategy.upper()} COMM AT EPOCH {end_epoch}')
                        self.communicate(
                            task_id, end_epoch, freq, num_epochs, strategy=strategy, final=final)

            if self.sharing_strategy.pre_or_post_comm == "post":
                for strategy, freq in comm_freqs.items():
                    if end_epoch % freq == 0 and freq <= num_epochs:
                        logging.info(
                            f'Task {task_id} {strategy.upper()} COMM AT EPOCH {end_epoch}')
                        self.communicate(
                            task_id, end_epoch, freq, num_epochs, strategy=strategy, final=final)

            start_epoch = end_epoch

    def communicate(self, task_id, end_epoch, comm_freq, num_epochs, start_com_round=0, final=False, strategy=None):
        for communication_round in range(start_com_round, self.num_coms_per_round + start_com_round):
            # parallelize preprocessing to prepare neccessary data
            # before the communication round.
            ray.get([agent.prepare_communicate.remote(task_id, end_epoch, comm_freq,
                                                      num_epochs,
                                                      communication_round, final, strategy)
                    for agent in self.agents])
            # NOTE: HACK: communicate in done in sequence to avoid dysnc issues,
            # if the sender sends something to the receiver but the receiver is not paying
            # attention, then the message is lost. Truly decen agent would have to have a
            # while True loop and implement some more complicated logic.
            for agent in self.agents:
                # sequential, because ray.get is blocking to get the result from one
                # agent before moving to the next.
                ray.get(agent.communicate.remote(
                    task_id, communication_round, final, strategy))

            ray.get([agent.process_communicate.remote(task_id, communication_round, final, strategy)
                    for agent in self.agents])

    def get_save_dir(self):
        return ray.get([agent.get_save_dir.remote() for agent in self.agents])

    def change_save_dir(self, save_dir):
        for agent in self.agents:
            node_id = ray.get(agent.get_node_id.remote())
            agent_save_dir = os.path.join(save_dir, f"agent_{node_id}")
            ray.get(agent.change_save_dir.remote(agent_save_dir))

    def load_records(self):
        ray.get([agent.load_records.remote() for agent in self.agents])
