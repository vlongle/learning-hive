'''
File: /gradient_fleet.py
Project: fleet
Created Date: Thursday March 23rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from copy import deepcopy
import logging
import ray
from shell.fleet.fleet import Fleet, ParallelFleet
import networkx as nx
from shell.fleet.utils.model_sharing_utils import exclude_model
import torch
import shutil


class SyncBaseFleet(Fleet):
    """
    TODO: rename GradFleet to something else.
    This class is used to train agents on the same initial tasks, and then proceed with individual local training.
    This is used for both sharing weights (grad) and modules (mod)
    """

    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs,
                 fake_dataset):
        # NOTE: We create a fake agent to jointly train all agents in the same initial tasks. Then,
        # we pretend that all agents have no initial tasks, and proceed with individual local training

        tmp_agent_kwargs = deepcopy(agent_kwargs)
        tmp_agent_kwargs["fl_strategy"] = None

        self.jointly_trained_agent = AgentCls(69420, seed, fake_dataset, NetCls, LearnerCls,
                                              deepcopy(net_kwargs), deepcopy(
                                                  tmp_agent_kwargs),
                                              deepcopy(train_kwargs), deepcopy(sharing_strategy))
        self.num_init_tasks = net_kwargs["num_init_tasks"]
        super().__init__(graph, seed, datasets, sharing_strategy, AgentCls,
                         NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs)

        self.uniformize_init_tasks()

    def uniformize_init_tasks(self, dataset=None):
        """
        """
        if dataset is None:
            dataset = self.jointly_trained_agent.dataset
        for agent in self.agents:
            for task in range(self.num_init_tasks):
                agent.replace_dataset(dataset, task)

    def train_and_comm(self, task_id):
        if task_id < self.num_init_tasks:
            # jointly train on the initial tasks
            self.jointly_trained_agent.train(task_id)
        else:
            if task_id == self.num_init_tasks:
                # now that we are done with joint training, we can delete the jointly_trained_agent
                self.copy_from_jointly_trained_agent()
                self.delete_jointly_trained_agent()
            return super(SyncBaseFleet, self).train_and_comm(task_id)

    def copy_from_jointly_trained_agent(self, net=None, dataset=None,
                                        record=None):
        """
        1. Copy the weights over
        2. Copy the replay buffer over
        3. Copy the logging to record.csv
        """
        if net is None:
            net = self.jointly_trained_agent.get_model()
        if dataset is None:
            dataset = self.jointly_trained_agent.dataset
        if record is None:
            record = self.jointly_trained_agent.get_record()

        for node in self.agents:
            node.replace_model(net)

        for task_id in range(self.num_init_tasks):
            train_loader = (
                torch.utils.data.DataLoader(dataset.trainset[task_id],
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True,
                                            ))

            for node in self.agents:
                node.replace_replay(train_loader, task_id)

        for node in self.agents:
            node.replace_record(record)

    def delete_jointly_trained_agent(self):
        # delete the self.jointly_trained_agent.save_dir folder
        # to prevent the statististics of this fake agent from being saved
        save_dir = self.jointly_trained_agent.get_save_dir()
        # delete the save_dir
        shutil.rmtree(save_dir)
        del self.jointly_trained_agent


class ParallelSyncBaseFleet(ParallelFleet):
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs,
                 fake_dataset):
        tmp_agent_kwargs = deepcopy(agent_kwargs)
        tmp_agent_kwargs["fl_strategy"] = None
        self.fake_dataset = fake_dataset

        self.jointly_trained_agent = AgentCls.options(num_gpus=1).remote(69420, seed, fake_dataset, NetCls, LearnerCls,
                                                                         deepcopy(net_kwargs), deepcopy(
                                                                             tmp_agent_kwargs),
                                                                         deepcopy(train_kwargs), deepcopy(sharing_strategy))
        self.num_init_tasks = net_kwargs["num_init_tasks"]
        self.args = (graph, seed, datasets, sharing_strategy, AgentCls,
                     NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs)

    def delete_jointly_trained_agent(self):
        # delete the self.jointly_trained_agent.save_dir folder
        # to prevent the statististics of this fake agent from being saved
        # save_dir = ray.get(self.jointly_trained_agent.get_save_dir.remote())
        # # delete the save_dir
        # shutil.rmtree(save_dir)
        ray.kill(self.jointly_trained_agent)
        del self.jointly_trained_agent

    def train_and_comm(self, task_id):
        if task_id < self.num_init_tasks:
            # jointly train on the initial tasks
            ray.get(self.jointly_trained_agent.train.remote(task_id))
        else:
            if task_id == self.num_init_tasks:
                net = ray.get(self.jointly_trained_agent.get_model.remote())
                record = ray.get(
                    self.jointly_trained_agent.get_record.remote())
                # now that we are done with joint training, we can delete the jointly_trained_agent
                # to free up GPU for the fleet
                self.delete_jointly_trained_agent()
                super().__init__(*self.args)
                self.uniformize_init_tasks(self.fake_dataset)
                self.copy_from_jointly_trained_agent(
                    net, self.fake_dataset, record)
            return super(ParallelSyncBaseFleet, self).train_and_comm(task_id)

    def copy_from_jointly_trained_agent(self, net, dataset, record):
        ray.get([
            agent.replace_model.remote(net) for agent in self.agents
        ])

        for task_id in range(self.num_init_tasks):
            train_loader = (
                torch.utils.data.DataLoader(dataset.trainset[task_id],
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True,
                                            ))
            ray.get([
                agent.replace_replay.remote(train_loader, task_id) for agent in self.agents
            ])

        ray.get([
            agent.replace_record.remote(record) for agent in self.agents
        ])

    def uniformize_init_tasks(self, dataset):
        for agent in self.agents:
            for task in range(self.num_init_tasks):
                ray.get(agent.replace_dataset.remote(deepcopy(dataset), task))
