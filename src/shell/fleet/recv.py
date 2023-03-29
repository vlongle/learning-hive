'''
File: /data_fleet.py
Project: fleet
Created Date: Thursday March 23rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import torch.nn.functional as F
import torch
import ray
from shell.fleet.fleet import Agent

"""
Receiver-first procedure.
Scorers return higher scores for more valuable instances (need to train more on).
"""


# ====================
# Unsupervised methods
# ====================
@torch.inference_mode()
def least_confidence_scorer(logits, labels=None):
    """
    Prioritize the samples with the lowest confidence (i.e., lowest maximum
    logits)
    """
    p = F.softmax(logits, dim=1)
    return -torch.max(p, dim=1).values


@torch.inference_mode()
def margin_scorer(logits, labels=None):
    """
    Prioritize the samples with the lowest margin (i.e., difference between
    two highest logits)
    """
    max_logits, _ = torch.max(logits, dim=1)
    second_max_logits = torch.topk(logits, k=2, dim=1).values[:, 1]
    return -(max_logits - second_max_logits)


@torch.inference_mode()
def entropy_scorer(logits, labels=None):
    """
    Prioritize the samples with the highest entropy (uncertainty
    of the softmax distribution)
    """
    probs = F.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs), dim=1)

# ====================
# Supervised Methods
# ====================


@torch.inference_mode()
def cross_entropy_scorer(logits, labels):
    """
    Prioritize the samples with the highest cross entropy loss
    """
    return F.cross_entropy(logits, labels, reduction='none')


class RecvDataAgent(Agent):
    """
    Have two rounds of communications.
    1. send query to neighbors
    2. send data to neighbors
    """

    def compute_ball(self, metric="cosine"):
        # get all the data from the replay buffer to compute the ball tree
        from sklearn.neighbors import BallTree
        for task_id in range(self.task_id):
            X, _, _ = self.replay_buffers[task_id]
            self.ball_trees[task_id] = BallTree(X, metric=metric)

    def nearest_neighbors(self, X, n_neighbors):
        """
        Get the nearest neighbors to X in the dataset.
        """
        # we have self.replay_buffers[tasks] = (X, y)
        task_neighbors = {}
        for task_id in range(self.task_id):
            # use self.ball_trees[task_id] to get the nearest neighbors
            # to X in the dataset
            dist, ind = self.ball_trees[task_id].query(
                X, k=n_neighbors)
            task_neighbors[task_id] = (dist, ind)
        return task_neighbors

    def compute_query(self, task_id):
        # get all valdataset from task=0 to task=task_id
        valdataset = torch.utils.data.ConcatDataset(
            self.dataset.valsets[:task_id+1])
        X_val, y_val = valdataset.tensors[0], valdataset.tensors[1]
        with torch.inference_mode():
            logits = self.net(X_val)
        if self.query_method == "unsupervised":
            scores = self.scorer(logits)
        elif self.query_method == "supervised":
            scores = self.scorer(logits, y_val)
        else:
            raise ValueError("Invalid query method")
        rank = torch.argsort(scores, descending=True)
        top_k = rank[:self.cfg.receiver_first.num_queries]
        top_k = top_k[scores[top_k] >
                      self.query_score_threshold]
        return X_val[top_k].cpu(), y_val[top_k].cpu()

    def receive(self, sender_id, data, data_type):
        if data_type == "query":
            # add query to buffer
            self.buffer_query[sender_id] = data
        elif data_type == "data":
            # add data to buffer
            self.buffer_data[sender_id] = data
        else:
            raise ValueError("Invalid data type")

    def remove_outliers(self):
        """
        TODO: 
        """
        pass

    def compute_data(self, query):
        # Get relevant data for the query
        pass

    def learn_from_buffer(self):
        # get the data and now learn from it
        pass

    def prepare_communicate(self, task_id, communication_round):
        if communication_round == 0:
            self.query = self.compute_query(task_id)

        for requester, query in self.buffer_query.items():
            data = ray.put(self.compute_data(query))
            requester.receive.remote(
                self.node_id, self.data[requester], "data")

    def communicate(self, task_id, communication_round):
        # https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/
        # storing query to the (node) object store so that other
        # processes can access it faster
        if communication_round == 0:
            # send query to neighbors
            for neighbor in self.neighbors:
                neighbor.receive.remote(self.node_id, self.query, "query")
        elif communication_round == 1:
            # send data to the requester
            for requester in self.buffer_query:
                requester.receive(
                    self.node_id, self.data[requester], "data")
        else:
            raise ValueError("Invalid round number")

    def process_communicate(self, task_id, communication_round):
        if communication_round == 0:
            self.compute_data()
        elif communication_round == 1:
            self.learn_from_buffer()
        else:
            raise ValueError("Invalid round number")
