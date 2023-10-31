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


class GradFleet(Fleet):
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


    def uniformize_init_tasks(self):
        """
        """
        for agent in self.agents:
            for task in range(self.num_init_tasks):
                agent.dataset.trainset[task] = self.jointly_trained_agent.dataset.trainset[task]
                agent.dataset.testset[task] = self.jointly_trained_agent.dataset.testset[task]
                agent.dataset.valset[task] = self.jointly_trained_agent.dataset.valset[task]
                agent.dataset.class_sequence[:task * (agent.dataset.num_classes_per_task)] = self.jointly_trained_agent.dataset.class_sequence[:task * (agent.dataset.num_classes_per_task)]


    def train_and_comm(self, task_id):
        if task_id < self.num_init_tasks:
            # jointly train on the initial tasks
            self.jointly_trained_agent.train(task_id)
        else:
            if task_id == self.num_init_tasks:
                # now that we are done with joint training, we can delete the jointly_trained_agent
                self.copy_from_jointly_trained_agent()
                del self.jointly_trained_agent
            return super(GradFleet, self).train_and_comm(task_id)
    
    def copy_from_jointly_trained_agent(self):
        """
        1. Copy the weights over
        2. Copy the replay buffer over
        3. Copy the logging to record.csv
        """
        for agent in self.agents:
            agent.net.load_state_dict(self.jointly_trained_agent.net.state_dict())

        for task_id in range(self.num_init_tasks):
            train_loader = (
            torch.utils.data.DataLoader(self.jointly_trained_agent.dataset.trainset[task_id],
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        ))

            for node in self.agents:
                node.agent.update_multitask_cost(train_loader, task_id)
                node.agent.T += 1
                node.agent.observed_tasks.add(task_id)

        for node in self.agents:
            node.agent.record.df = self.jointly_trained_agent.agent.record.df.copy()
            # overwrite the record.csv file
            node.agent.record.save()



class ParallelGradFleet(GradFleet, ParallelFleet):
    pass
