'''
File: /sender.py
Project: fleet
Created Date: Tuesday March 28th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.fleet.fleet import Agent
import torch.nn as nn
import torch


class SendDataAgent(Agent):
    """
    Even round: send data.
    Odd round: send feedback.
    """

    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
        # remember which data is already received by the neighbors
        # already_taken[neighbor][task] = set of idx of data already taken
        self.already_taken = {
            neighbor: {task: set() for task in range(self.net.num_tasks)} for neighbor in self.neighbors}

    def prepare_communicate(self, task_id, communication_round):
        if communication_round == 0:
            # create a bunch of regressors for my neighbors
            self.regressors = {}
            self.eps = {}
            self.regress_optimizers = {}
            for neighbor in self.neighbors:
                self.regressors[neighbor] = nn.Linear(
                    self.net.get_hidden_size(), 1)
                self.eps[neighbor] = self.sharing_strategy.init_eps
                self.regress_optimizers[neighbor] = torch.optim.Adam(
                    self.regressors[neighbor].parameters())
        if communication_round % 2 == 0:
            # greedy-eps to prepare data to send
            self.prepare_data()

    def communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            # send data to neighbors
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.data, "data")
        else:
            # send feedback to neighbors
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.feedback, "feedback")

    def get_candidate_data(self, task_id, neighbor_id):
        # get the data that is not already taken by the neighbor
        # return a list of indices
        data = self.replay_buffer[task_id]
        # get the indices of the data
        indices = list(range(len(data)))
        # get the indices that are not already taken
        candidate_indices = list(
            set(indices) - self.already_taken[neighbor_id][task_id])
        return data[candidate_indices], candidate_indices

    def prepare_data(self, task_id, neighbor_id):
        """
        Get data from all task <= task_id from 
        self.get_candidate_data. Compute the
        predicted preference scores from self.regressors[neighbor_id].
        Then run self.select() function (which internally uses eps-greedy)
        to select self.sharing_strategy.bandwidth data to send to the neighbor.
        and remember the candidate_idx being selected.
        Data is in the form
        {
            "task_id": task_id,
            "class": class,
            "data": data,
        }
        """
        # get data from all task <= task_id
        data = []
        candidate_indices = []
        for task in range(task_id):
            d, idx = self.get_candidate_data(task, neighbor_id)
            data.append(d)
            candidate_indices.append(idx)
        # compute the predicted preference scores
        data = torch.cat(data)
        scores = self.regressors[neighbor_id](data)

    def evaluate_data(self):
        pass

    def update_engine(self):
        pass

    def process_communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            self.evaluate_data()
        else:
            self.update_engine()
