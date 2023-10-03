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
from torchmetrics.functional import pairwise_cosine_similarity
from shell.utils.replay_buffers import ReplayBufferReservoir

"""
Receiver-first procedure.
Scorers return higher scores for more valuable instances (need to train more on).
"""

"""
TODO:
1) return debug stuff.
2) maybe thresholding to throw away trash
3) Use original source labels and just learn contrastive.


4) Maybe BUG??? compute_raw_dist should exactly be the same given the same query,
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
    def compute_embedding_dist(self, X1, X2, task_id):
        self.net.eval()
        X1_embed = self.net.encode(X1.to(self.net.device), task_id=task_id) # (B, hidden_dim)
        X2_embed = self.net.encode(X2.to(self.net.device), task_id=task_id)
        sim = pairwise_cosine_similarity(X1_embed, X2_embed)
        return sim.cpu()

    # @torch.inference_mode()
    # def compute_embedding_dist(self, X1, X2, task_id):
    #     self.net.eval()
    #     X1_embed = self.net.encode(X1.to(self.net.device), task_id=task_id) # (B, hidden_dim)
    #     X2_embed = self.net.encode(X2.to(self.net.device), task_id=task_id)
    #     sim = torch.cdist(X1_embed, X2_embed)
    #     return sim.cpu()

    def compute_raw_dist(self, X1, X2, task_id=None):
        # make sure X1.shape == X2.shape
        # if X1.shape is nnnot [N, d] then flatten it
        if len(X1.shape) > 2:
            X1 = X1.reshape(X1.shape[0], -1)
            X2 = X2.reshape(X2.shape[0], -1)
        sim = pairwise_cosine_similarity(X1, X2)
        return sim
    
    # def compute_raw_dist(self, X1, X2, task_id=None):
    #     # make sure X1.shape == X2.shape
    #     # if X1.shape is not [N, d] then flatten it
    #     if len(X1.shape) > 2:
    #         X1 = X1.reshape(X1.shape[0], -1)
    #         X2 = X2.reshape(X2.shape[0], -1)
        
    #     # Compute the L2 distance
    #     dist = torch.cdist(X1, X2)
    #     return dist

    @torch.inference_mode()
    def compute_similarity(self, qX, computer=None, chosen_tasks=None):
        """
        Loop through all the data of each task that we currently have
        and compute the similarity with qX. Return the considered
        data as a concat X, Y, and task tensors, and similarity scores.

        For current task, consider the training set. For previous task,
        we don't have access to the training data anymore only the replay
        buffer.
        """
        if computer is None:
            computer = self.compute_embedding_dist

        sims = []
        Xs, ys, tasks = [], [], []
        
        replay_buffers = self.agent.replay_buffers
        if chosen_tasks is not None:
            replay_buffers = {t: replay_buffers[t] for t in chosen_tasks}

        for t, replay in replay_buffers.items():
            if t == self.agent.T - 1:
                # current task
                Xt, yt = self.dataset.trainset[t].tensors
            else:
                Xt = replay.tensors[0]
                yt = replay.tensors[1]
            sim = computer(qX, Xt, t)
            sims.append(sim)
            Xs.append(Xt)
            ys.append(yt)
            tasks.append(torch.ones(Xt.shape[0], dtype=torch.long) * t)

        sims = torch.cat(sims, dim=1)
        Xs = torch.cat(Xs, dim=0)
        ys = torch.cat(ys, dim=0)
        tasks = torch.cat(tasks, dim=0)

        return sims, Xs, ys, tasks

    @torch.inference_mode()
    def extract_topk_from_similarity(self, sims, Xs, ys, tasks, neighbors, candidate_tasks=None):
        """
        candidate_tasks.shape = [N, n_filter_neighbors] where N is the number of
        query points and n_filter_neighbors is the number of neighbors to consider

        Xs.shape = [N, ...]

        tasks.shape = [M] where M is the number of candidate data points (e.g., data from various
        task buffers or the current training task)

        sims.shape = [N, M] where sims[i, j] is the similarity between query point i
        and data point j
        """
        if candidate_tasks is not None:
            # Step 1: Reshape task_neighbors_prefilter
            expanded_prefilter = candidate_tasks.unsqueeze(-1) # Shape: [N, n_filter_neighbors, 1]

            # Step 2: Expand tasks
            expanded_tasks = tasks.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, M]

            # Step 3: Use broadcasting to compare the two tensors
            comparison_result = expanded_prefilter == expanded_tasks # Shape: [N, n_filter_neighbors, M]

            # Step 4: Aggregate along the n_filter_neighbors dimension
            mask = comparison_result.any(dim=1).float() # Shape: [N, M], valid task is 1, invalid task is 0

            # Step 5: articially lower the score for data from non-candidate tasks
            invalid_mask = (1 - mask).bool()
            sims = torch.where(invalid_mask, torch.tensor(-float('inf'), device=sims.device), sims)




        top_k = torch.topk(sims, k=neighbors, dim=1).indices

        X_neighbors = Xs[top_k.flatten()]
        Y_neighbors = ys[top_k.flatten()]
        task_neighbors = tasks[top_k.flatten()]

        # reshape to (N, n_neighbor, c, h, w)
        X_neighbors = X_neighbors.reshape(
            sims.shape[0], neighbors, *Xs.shape[1:])
        Y_neighbors = Y_neighbors.reshape(sims.shape[0], neighbors)
        task_neighbors = task_neighbors.reshape(sims.shape[0], neighbors)

        return X_neighbors, Y_neighbors, task_neighbors

    def new_nearest_neighbors(self, qX, n_neighbors, n_filter_neighbors,
                              debug=False):
        """
        Pre-filter with raw pixel comparison and then refine with embeddings.
        """
        # 1. Pre-filter using raw pixel comparison
        sims_prefilter, Xs, ys, tasks = self.compute_similarity(
            qX, 
            computer=self.compute_raw_dist
        )
        # print("tasks.shape:", tasks.shape, "Xs:", Xs.shape, "ys:", ys.shape, "sims_prefilter:", sims_prefilter.shape)
        X_n_prefilter, y_n_prefilter, task_neighbors_prefilter = self.extract_topk_from_similarity(
            sims_prefilter, Xs, ys, tasks, 
            neighbors=n_filter_neighbors
        )

        # Convert to list of task indices for each query point
        # task_lists = [tasks[indices].tolist() for indices in task_neighbors_prefilter]

        # 2. Compute similarity using embedding method
        sims, _, _, _ = self.compute_similarity(qX, computer=self.compute_embedding_dist)

        # 3. Extract top neighbors considering the pre-filtered tasks
        X_neighbors, _, _ = self.extract_topk_from_similarity(
            sims, Xs, ys, tasks, 
            neighbors=n_neighbors, 
            # candidate_tasks=task_lists,
            candidate_tasks=task_neighbors_prefilter,
        )
        
        if debug:
            return X_neighbors, X_n_prefilter
        return X_neighbors





    def nearest_neighbors(self, qX, neighbors: int, computer=None, debug=False, chosen_tasks=None):
        """
        NOTE: old algorithm without the pre-filtering step. Kept here for reference.


        
        First, go through every task in the replay buffer and compute
        the cosine similarity scores between X and the data in the replay buffer.

        At the end, take the top `neighbors` number of data points


        X.shape = (N, C, H, W)

        Return: X_neighbors of shape (N, n_neighbor, C, H, W)
        """
        if computer is None:
            computer = self.compute_embedding_dist

        sims = []
        Xs, ys, tasks = [], [], []
        was_training = self.net.training
        qX = qX.to(self.net.device)



        replay_buffers = self.agent.replay_buffers
        if chosen_tasks is not None:
            replay_buffers = {t: replay_buffers[t] for t in chosen_tasks}

        for t, replay in replay_buffers.items():
            Xt = replay.tensors[0]
            yt = replay.tensors[1]
            sim = computer(qX, Xt, t)
            sims.append(sim)
            Xs.append(Xt)
            ys.append(yt)
            tasks.append(torch.ones(Xt.shape[0], dtype=torch.long) * t)

        sims = torch.cat(sims, dim=1)
        # print('sims:', sims.shape)
        Xs = torch.cat(Xs, dim=0)
        ys = torch.cat(ys, dim=0)
        tasks = torch.cat(tasks, dim=0)

        top_k = torch.topk(sims, k=neighbors, dim=1).indices
        X_neighbors = Xs[top_k.flatten()]
        Y_neighbors = ys[top_k.flatten()]
        task_neighbors = tasks[top_k.flatten()]

        # reshape to (N, n_neighbor, c, h, w)
        X_neighbors = X_neighbors.reshape(
            qX.shape[0], neighbors, *qX.shape[1:])
        Y_neighbors = Y_neighbors.reshape(
            qX.shape[0], neighbors)
        task_neighbors = task_neighbors.reshape(
            qX.shape[0], neighbors)
        if was_training:
            self.net.train()
        
        if debug:
            return X_neighbors, Y_neighbors, task_neighbors
        return X_neighbors

    def get_valset(self, tasks):
        # NOTE: TODO: probably should get the query from the replay buffer instead
        X_vals = {
            t: self.dataset.valset[t].tensors[0]
            for t in tasks
        }
        y_vals = {
            t: self.dataset.valset[t].tensors[1]
            for t in tasks
        }
        return X_vals, y_vals

    @torch.inference_mode()
    def compute_query(self, task_id, mode="all", debug_return=False):
        """
        Compute query using a validation

        If mode="all", get the query for all tasks up to `task_id`,
        If mode="current", get the query for the current task only.

        TODO: NOTE: right now, each task gets equal no. of queries but
        we might want to make that dependent on the actual score.
        """

        was_training = self.net.training
        if mode == "all":
            tasks = range(task_id + 1)
        elif mode == "current":
            tasks = [task_id]
        else:
            raise ValueError(f"Invalid mode {mode}")

        X_queries = {}
        y_queries = {}
        y_pred_queries = {}
        score_queries = {}

        X_vals, y_vals = self.get_valset(tasks)

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
        return X_queries, y_queries

    


    def receive(self, sender_id, data, data_type):
        if data_type == "query":
            # add query to buffer
            self.incoming_query[sender_id] = data
        elif data_type == "data":
            # add data to buffer
            self.incoming_data[sender_id] = data
        else:
            raise ValueError("Invalid data type")


    def compute_data(self):
        # Get relevant data for the query
        # and populate self.data
        self.data = {}
        for requester, query in self.incoming_query.items():
            # query is a dict of task_id -> X
            concat_query = torch.cat(list(query.values()), dim=0) # concat_query = size (N, C, H, W)
            data = self.new_nearest_neighbors(concat_query,
                                             self.sharing_strategy.num_data_neighbors,
                                             self.sharing_strategy.num_filter_neighbors)
            # data size = (N, n_neighbor, C, H, W)
            # put the data back into a dict
            structured_data = {}
            start_idx = 0
            for task_id, X in query.items():
                end_idx = start_idx + X.size(0)  # Assuming the 0th dimension is the size N
                structured_data[task_id] = data[start_idx:end_idx]
                start_idx = end_idx  # Update start_idx for the next iteration

            self.data[requester] = structured_data
            

    def add_incoming_data(self):
        # get the data and now learn from it
        for neighbor_id, neighbor_data in self.incoming_data.items():
            for task_id, task_data in neighbor_data.items():
                # task_data.shape = (N, n_neighbor, C, H, W)
                if task_id not in self.agent.shared_replay_buffers:
                    self.agent.shared_replay_buffers[task_id] = ReplayBufferReservoir(
                        self.sharing_strategy.shared_memory_size, task_id)
                Y = self.query_y[task_id]  # Y.shape = (N)
                n_neighbor = task_data.shape[1]  # Extracting n_neighbor from task_data shape
                
                # Expanding Y to shape (N, n_neighbor) 
                # and then flattening it to (N*n_neighbor,)
                Y_expanded = Y.unsqueeze(1).expand(-1, n_neighbor).reshape(-1)
                
                # Flattening task_data to (N*n_neighbor, C, H, W)
                X_flattened = task_data.reshape(-1, *task_data.shape[2:])
                
                # Storing flattened X and Y into the replay buffer
                self.agent.shared_replay_buffers[task_id].push(X_flattened, Y_expanded)
        
        ## TODO: start learning now!
    

    def prepare_communicate(self, task_id, communication_round):
        if communication_round == 0:
            self.incoming_query, self.incoming_data = {}, {}
        if task_id < self.agent.net.num_init_tasks - 1:
            return
        if communication_round == 0:
            X, y = self.compute_query(task_id)
            self.query = X
            self.query_y = y
        elif communication_round == 1:
            self.compute_data()
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    # potentially parallelizable
    def communicate(self, task_id, communication_round):
        if task_id < self.agent.net.num_init_tasks - 1:
            # NOTE: don't communicate for the first few tasks to
            # allow agents some initital training to find their weakness
            return
        if communication_round == 0:
            # send query to neighbors
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.query, "query")
        elif communication_round == 1:
            # send data to the requester
            # for requester in self.incoming_query:
            #     self.neighbors[requester].receive(
            #         self.node_id, self.data[requester], "data")
            for neighbor in self.neighbors:
                neighbor.receive(
                    self.node_id, self.data[neighbor.node_id], "data")
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    def process_communicate(self, task_id, communication_round):
        if task_id < self.agent.net.num_init_tasks - 1:
            return
        if communication_round == 0:
            pass
        elif communication_round == 1:
            self.add_incoming_data()
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
            for requester in self.incoming_query:
                requester.remote.receive(
                    self.node_id, self.data[requester], "data")
        else:
            raise ValueError(f"Invalid round number {communication_round}")
