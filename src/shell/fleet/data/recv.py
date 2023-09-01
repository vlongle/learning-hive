'''
File: /data_fleet.py
Project: fleet
Created Date: Thursday March 23rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


from sklearn.neighbors import BallTree
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
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.sum(probs * log_probs, dim=1)
    # return -torch.sum(probs * torch.log(probs), dim=1)

# ====================
# Supervised Methods
# ====================


@torch.inference_mode()
def cross_entropy_scorer(logits, labels):
    """
    Prioritize the samples with the highest cross entropy loss
    """
    return F.cross_entropy(logits, labels, reduction='none')


SCORER_FN_LOOKUP = {
    "least_confidence": least_confidence_scorer,
    "margin": margin_scorer,
    "entropy": entropy_scorer,
    "cross_entropy": cross_entropy_scorer,
}

SCORER_TYPE_LOOKUP = {
    "least_confidence": "unsupervised",
    "margin": "unsupervised",
    "entropy": "unsupervised",
    "cross_entropy": "supervised",
}


class RecvDataAgent(Agent):
    """
    Have two rounds of communications.
    1. send query to neighbors
    2. send data to neighbors
    """

    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs,
                 sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.scorer = SCORER_FN_LOOKUP[self.sharing_strategy.scorer]
        self.scorer_type = SCORER_TYPE_LOOKUP[self.sharing_strategy.scorer]

    @torch.inference_mode()
    def compute_ball(self, metric="cosine"):
        was_training = self.net.training
        self.net.eval()
        self.ball_trees = {}
        # get all the data from the replay buffer to compute the ball tree
        for t, replay in self.agent.replay_buffers.items():
            X = replay.tensors[0]  # (B, C, H, W)
            X_embed = self.net.encode(X, task_id=t)  # (B, hidden_dim)
            self.ball_trees[t] = BallTree(X_embed, metric=metric)
        if was_training:
            self.net.train()

    def nearest_neighbors(self, X, n_neighbors: int):
        """
        Get the nearest neighbors to X in the dataset.
        """
        # we have self.replay_buffers[tasks] = (X, y)
        task_neighbors = {}
        for t, ball_tree in self.ball_trees.items():
            # use self.ball_trees[task_id] to get the nearest neighbors
            # to X in the dataset
            dist, ind = ball_tree.query(
                X, k=n_neighbors)
            task_neighbors[t] = (dist, ind)
        return task_neighbors

    @torch.inference_mode()
    def compute_query(self, task_id, mode="all", debug_return=False):
        """
        Compute query using a validation

        If mode="all", get the query for all tasks up to `task_id`,
        If mode="current", get the query for the current task only.
        """
        was_training = self.net.training
        if mode == "all":
            tasks = range(task_id + 1)
        elif mode == "current":
            tasks = [task_id]
        else:
            raise ValueError(f"Invalid mode {mode}")

        X_vals = {
            t: self.dataset.valset[t].tensors[0]
            for t in tasks
        }
        y_vals = {
            t: self.dataset.valset[t].tensors[1]
            for t in tasks
        }
        X_queries = {}
        y_queries = {}
        y_pred_queries = {}
        score_queries = {}
        with torch.inference_mode():
            for t in tasks:
                X_val = X_vals[t].to(self.net.device)
                y_val = y_vals[t].to(self.net.device)
                logits = self.net(X_val, task_id=t)
                y_pred = torch.argmax(logits, dim=1)
                if self.scorer_type == "unsupervised":
                    scores = self.scorer(logits)
                elif self.scorer_type == "supervised":
                    scores = self.scorer(logits, y_val)
                else:
                    raise ValueError("Invalid query method")
                rank = torch.argsort(scores, descending=True)
                top_k = rank[:self.sharing_strategy.num_queries]
                top_k = top_k[scores[top_k] >
                              self.sharing_strategy.query_score_threshold]
                X_queries[t] = X_val[top_k].cpu()
                y_queries[t] = y_val[top_k].cpu()
                y_pred_queries[t] = y_pred[top_k].cpu()
                score_queries[t] = scores[top_k].cpu()

        if was_training:
            self.net.train()
        if debug_return:
            return X_queries, y_queries, y_pred_queries, score_queries
        return X_queries

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
        pass

    def compute_data(self, query):
        # Get relevant data for the query
        pass

    def learn_from_recv_data(self):
        # get the data and now learn from it
        pass

    def prepare_communicate(self, task_id, communication_round):
        if communication_round == 0:
            self.query = self.compute_query(task_id)
        elif communication_round == 1:
            self.compute_data()
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    # potentially parallelizable
    def communicate(self, task_id, communication_round):
        if communication_round == 0:
            # send query to neighbors
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.query, "query")
        elif communication_round == 1:
            # send data to the requester
            for requester in self.buffer_query:
                requester.receive(
                    self.node_id, self.data[requester], "data")
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    def process_communicate(self, task_id, communication_round):
        if communication_round == 0:
            pass
        elif communication_round == 1:
            self.learn_from_recv_data()
        else:
            raise ValueError("Invalid round number")


class ParallelRecvDataAgent(RecvDataAgent):
    def communicate(self, task_id, communication_round):
        if communication_round == 0:
            # send query to neighbors
            for neighbor in self.neighbors:
                neighbor.remote.receive(self.node_id, self.query, "query")
        elif communication_round == 1:
            # send data to the requester
            for requester in self.buffer_query:
                requester.remote.receive(
                    self.node_id, self.data[requester], "data")
        else:
            raise ValueError(f"Invalid round number {communication_round}")
