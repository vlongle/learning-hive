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
from shell.fleet.data.data_utilize import *
import pickle
from shell.learners.base_learning_classes import CompositionalDynamicLearner
from functools import partial
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


@torch.inference_mode()
def random_scorer(logits, labels=None):
    """
    Return random scores
    """
    return torch.rand(logits.shape[0])

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
    "random": random_scorer,
}

SCORER_TYPE_LOOKUP = {
    "least_confidence": "unsupervised",
    "margin": "unsupervised",
    "entropy": "unsupervised",
    "cross_entropy": "supervised",
    "random": "unsupervised",
}


# @torch.inference_mode()
# def compute_embedding_dist(net, X1, X2=None, task_id=None):
#     assert task_id is not None
#     was_training = net.training
#     net.eval()
#     X1_embed = net.encode(
#         X1.to(net.device), task_id=task_id)  # (B, hidden_dim)
#     if X2 is not None:
#         X2_embed = net.encode(X2.to(net.device), task_id=task_id)
#         sim = pairwise_cosine_similarity(X1_embed, X2_embed)
#     else:
#         sim = pairwise_cosine_similarity(X1_embed)
#     if was_training:
#         net.train()
#     return sim.cpu()


@torch.inference_mode()
def compute_embedding_dist(net, X1, X2, task_id):
    net.eval()
    X1_embed = net.encode(
        X1.to(net.device), task_id=task_id)  # (B, hidden_dim)
    X2_embed =net.encode(X2.to(net.device), task_id=task_id)
    sim = pairwise_cosine_similarity(X1_embed, X2_embed)
    return sim.cpu()



class RecvDataAgent(Agent):
    """
    Have two rounds of communications.
    1. send query to neighbors
    2. send data to neighbors
    """

    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs,
                 sharing_strategy):
        # self.use_ood_separation_loss = sharing_strategy.use_ood_separation_loss
        # agent_kwargs['use_ood_separation_loss'] = self.use_ood_separation_loss
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.scorer = SCORER_FN_LOOKUP[self.sharing_strategy.scorer]
        self.scorer_type = SCORER_TYPE_LOOKUP[self.sharing_strategy.scorer]

        self.is_modular = isinstance(self.agent, CompositionalDynamicLearner)

    # @torch.inference_mode()
    # def compute_embedding_dist(self, X1, X2, task_id):
    #     self.net.eval()
    #     X1_embed = self.net.encode(
    #         X1.to(self.net.device), task_id=task_id)  # (B, hidden_dim)
    #     X2_embed = self.net.encode(X2.to(self.net.device), task_id=task_id)
    #     sim = torch.cdist(X1_embed, X2_embed)
    #     return 1.0 - sim.cpu()

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

    def get_candidate_data(self, task):
        if task == self.agent.T - 1:
            # current task
            Xt, yt = self.dataset.trainset[task].tensors
        else:
            Xt = self.agent.replay_buffers[task].tensors[0]
            yt = self.agent.replay_buffers[task].tensors[1]
        return Xt, yt

    @torch.inference_mode()
    def compute_similarity(self, qX, computer=None):
        """
        Loop through all the data of each task that we currently have
        and compute the similarity with qX. Return the considered
        data as a concat X, Y, and task tensors, and similarity scores.

        For current task, consider the training set. For previous task,
        we don't have access to the training data anymore only the replay
        buffer.
        """
        if computer is None:
            computer = partial(compute_embedding_dist, self.net)

        sims = []
        Xs, ys, tasks = [], [], []

        for t in range(self.agent.T):
            Xt, yt = self.get_candidate_data(t)
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
    def extract_topk_from_similarity(self, sims, Xs, ys, tasks, num_neighbors, candidate_tasks=None,
                                     map_to_globals=True):
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
            # Shape: [N, n_filter_neighbors, 1]
            expanded_prefilter = candidate_tasks.unsqueeze(-1)

            # Step 2: Expand tasks
            expanded_tasks = tasks.unsqueeze(
                0).unsqueeze(0)  # Shape: [1, 1, M]

            # Step 3: Use broadcasting to compare the two tensors
            # Shape: [N, n_filter_neighbors, M]
            comparison_result = expanded_prefilter == expanded_tasks

            # Step 4: Aggregate along the n_filter_neighbors dimension
            # Shape: [N, M], valid task is 1, invalid task is 0
            mask = comparison_result.any(dim=1).float()

            # Step 5: artificially lower the score for data from non-candidate tasks
            invalid_mask = (1 - mask).bool()
            sims = torch.where(
                invalid_mask, torch.tensor(-float('inf'), device=sims.device), sims)

        top_k = torch.topk(sims, k=num_neighbors, dim=1)
        top_k_indices = top_k.indices
        top_k_values = top_k.values

        if map_to_globals:
            # first convert ys to global labels using tasks
            ys = get_global_labels(
                ys, tasks, self.dataset.class_sequence, self.dataset.num_classes_per_task)

        X_neighbors = Xs[top_k_indices.flatten()]
        Y_neighbors = ys[top_k_indices.flatten()]
        task_neighbors = tasks[top_k_indices.flatten()]
        sim_neighbors = top_k_values.flatten()

        # reshape to (N, n_neighbor, c, h, w)
        X_neighbors = X_neighbors.reshape(
            sims.shape[0], num_neighbors, *Xs.shape[1:])
        Y_neighbors = Y_neighbors.reshape(sims.shape[0], num_neighbors)
        task_neighbors = task_neighbors.reshape(sims.shape[0], num_neighbors)
        sim_neighbors = sim_neighbors.reshape(sims.shape[0], num_neighbors)

        return X_neighbors, Y_neighbors, task_neighbors, sim_neighbors

    def prefilter(self, qX, neighbor_id, n_filter_neighbors):
        if self.sharing_strategy['prefilter_strategy'] == 'raw_distance':
            return self.prefilter_raw_distance(qX, n_filter_neighbors)
        elif self.sharing_strategy['prefilter_strategy'] == 'none':
            return self.prefilter_none(qX, n_filter_neighbors)
        elif self.sharing_strategy['prefilter_strategy'] == 'oracle':
            return self.prefilter_oracle(qX, neighbor_id, n_filter_neighbors)
        else:
            raise ValueError(
                f"Invalid prefilter strategy {self.sharing_strategy['prefilter_strategy']}")

    def prefilter_none(self, qX, n_filter_neighbors):
        return {
            "task_neighbors_prefilter": None,
        }

    def prefilter_oracle(self, qX, neighbor_id, n_filter_neighbors):
        # dict of task_id -> global_y
        query_global_y = self.incoming_query_extra_info[neighbor_id]['query_global_y']
        query_global_y = torch.cat(
            list(query_global_y.values()), dim=0)  # shape=(num_queries)
        # print('query_global_y', query_global_y)

        return {
            "task_neighbors_prefilter": self.prefilter_oracle_helper(qX, query_global_y, n_filter_neighbors)
        }

    # NOTE: might not get all possible tasks...
    def prefilter_oracle_helper_legacy(self, qX, q_global_Y, n_filter_neighbors):
        assert q_global_Y.shape[0] == qX.shape[0]
        _, task_ids = get_local_labels(
            q_global_Y, self.dataset.class_sequence, self.dataset.num_classes_per_task)
        ret = torch.full(
            (q_global_Y.size(0), n_filter_neighbors), -1, dtype=torch.long)
        for i, task_id in enumerate(task_ids):
            if task_id == -1 or task_id >= self.agent.T:
                continue
            else:
                ret[i, :] = torch.full(
                    (n_filter_neighbors,), task_id, dtype=torch.long)
        return ret

    def prefilter_oracle_helper(self, qX, q_global_Y, n_filter_neighbors):
        assert q_global_Y.shape[0] == qX.shape[0]
        local_ys, task_ids_list = get_all_local_labels(
            q_global_Y, self.dataset.class_sequence, self.dataset.num_classes_per_task)
        ret = torch.full(
            (q_global_Y.size(0), n_filter_neighbors), -1, dtype=torch.long)

        for i, task_ids in enumerate(task_ids_list):
            # Filter out task IDs that exceed the current time horizon
            valid_task_ids = [
                task_id for task_id in task_ids if task_id < self.agent.T]

            n_valid_tasks = len(valid_task_ids)
            if n_valid_tasks == 0:
                continue
            else:
                # Fill with valid tasks and replicate the last valid task if necessary
                ret[i, :n_valid_tasks] = torch.tensor(
                    valid_task_ids, dtype=torch.long)
                if n_valid_tasks < n_filter_neighbors:
                    ret[i, n_valid_tasks:] = torch.full(
                        (n_filter_neighbors - n_valid_tasks,), valid_task_ids[-1], dtype=torch.long)
        return ret

    def prefilter_raw_distance(self, qX, n_filter_neighbors):
        # 1. Pre-filter using raw pixel comparison
        sims_prefilter, Xs, ys, tasks = self.compute_similarity(
            qX,
            computer=self.compute_raw_dist
        )
        # print("tasks.shape:", tasks.shape, "Xs:", Xs.shape, "ys:", ys.shape, "sims_prefilter:", sims_prefilter.shape)
        X_n_prefilter, y_n_prefilter, task_neighbors_prefilter = self.extract_topk_from_similarity(
            sims_prefilter, Xs, ys, tasks,
            num_neighbors=n_filter_neighbors
        )
        return {
            "X_n_prefilter": X_n_prefilter,
            "y_n_prefilter": y_n_prefilter,
            "task_neighbors_prefilter": task_neighbors_prefilter,
        }

    def new_nearest_neighbors(self, qX, neighbor_id, n_neighbors, n_filter_neighbors,
                              debug=False):
        """
        qX is a concatenated tensor of query points from all tasks.
        Pre-filter with raw pixel comparison and then refine with embeddings.
        """

        # Convert to list of task indices for each query point
        # task_lists = [tasks[indices].tolist() for indices in task_neighbors_prefilter]
        prefilter_info = self.prefilter(qX, neighbor_id, n_filter_neighbors)

        # 2. Compute similarity using embedding method
        sims, Xs, ys, tasks = self.compute_similarity(
            qX, computer=partial(compute_embedding_dist, self.net))

        # 3. Extract top neighbors considering the pre-filtered tasks
        X_neighbors, Y_neighbors, task_neighbors, sims = self.extract_topk_from_similarity(
            sims, Xs, ys, tasks,
            num_neighbors=n_neighbors,
            # candidate_tasks=task_lists,
            candidate_tasks=prefilter_info['task_neighbors_prefilter'],
        )

        if debug:
            return {
                "X_neighbors": X_neighbors,
                "Y_neighbors": Y_neighbors,
                "task_neighbors": task_neighbors,
                "sims": sims,
            } | prefilter_info
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

        Rank all instances across all tasks together based on their scores,
        and then select the top `num_queries` instances in a vectorized manner.
        """

        was_training = self.net.training
        self.net.eval()

        if mode == "all":
            tasks = range(task_id + 1)
        elif mode == "current":
            tasks = [task_id]
        else:
            raise ValueError(f"Invalid mode {mode}")

        X_vals, y_vals = self.get_valset(tasks)

        all_scores = []
        all_y_pred = []
        all_indices = {}
        cumulative_instance_count = 0

        # Collect scores and indices for all tasks
        for t in tasks:
            X_val = X_vals[t].to(self.net.device)
            y_val = y_vals[t].to(self.net.device)
            logits = self.net(X_val, task_id=t)
            y_pred = torch.argmax(logits, dim=1)
            scores = self.scorer(
                logits) if self.scorer_type == "unsupervised" else self.scorer(logits, y_val)
            all_scores.append(scores)
            all_y_pred.append(y_pred)
            all_indices[t] = torch.tensor([i for i in range(
                cumulative_instance_count, scores.size(0) + cumulative_instance_count)])
            cumulative_instance_count += scores.size(0)

        # Concatenate and rank all scores
        all_scores = torch.cat(all_scores)
        all_y_pred = torch.cat(all_y_pred)
        rank = torch.argsort(all_scores, descending=True)

        # Select the top k scores across all tasks
        top_k_indices = rank[:self.sharing_strategy.num_queries].cpu()

        X_queries = {t: torch.tensor([], dtype=torch.float32) for t in tasks}
        y_queries = {t: torch.tensor([], dtype=torch.int64) for t in tasks}
        y_pred_queries = {t: torch.tensor(
            [], dtype=torch.int64) for t in tasks}
        score_queries = {t: torch.tensor(
            [], dtype=torch.float32) for t in tasks}
        # print('top_k_indices', top_k_indices)
        # print('top_k_scores', all_scores[top_k_indices])

        # Vectorized approach to distribute queries back to their respective tasks
        for t in tasks:
            task_indices = all_indices[t]
            # Check if any of these indices are in the top k
            mask = torch.isin(task_indices, top_k_indices)

            # selected_indices = task_indices[mask]

            # If there are matches, process them
            if mask.any():
                # Get the global indices for the current task
                selected_global_indices = task_indices[mask]

                selected_scores = all_scores[selected_global_indices]
                sorted_indices = selected_global_indices[torch.argsort(
                    selected_scores, descending=True).cpu()]

                # Convert global indices to local indices
                # The starting index of the current task in the global indexing
                task_start_index = task_indices[0].item()
                selected_local_indices = sorted_indices - task_start_index

                # Retrieve the corresponding data using local indices
                X_queries[t] = X_vals[t][selected_local_indices].cpu()
                y_queries[t] = y_vals[t][selected_local_indices].cpu()
                y_pred_queries[t] = all_y_pred[sorted_indices].cpu()
                score_queries[t] = all_scores[sorted_indices].cpu()

        if was_training:
            self.net.train()
        if debug_return:
            return X_queries, y_queries, y_pred_queries, score_queries
        return X_queries, y_queries

    @torch.inference_mode()
    def compute_query_legacy(self, task_id, mode="all", debug_return=False):
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
        elif data_type == "extra_info":
            self.incoming_extra_info[sender_id] = data
        elif data_type == "query_extra_info":
            self.incoming_query_extra_info[sender_id] = data
        else:
            raise ValueError("Invalid data type")

    def helper_chunk(self, tensor, keys, lengths):
        structured_data = {}
        start_idx = 0
        for k, l in zip(keys, lengths):
            end_idx = start_idx + l
            structured_data[k] = tensor[start_idx:end_idx]
            start_idx = end_idx
        return structured_data

    def compute_data(self):
        # Get relevant data for the query
        # and populate self.data
        self.data = {}
        self.extra_info = {}
        for requester, query in self.incoming_query.items():
            # query is a dict of task_id -> X
            # concat_query = size (N, C, H, W)
            concat_query = torch.cat(list(query.values()), dim=0)
            data = self.new_nearest_neighbors(concat_query,
                                              requester,
                                              self.sharing_strategy.num_data_neighbors,
                                              self.sharing_strategy.num_filter_neighbors,
                                              debug=True)
            structured_data = {}
            # lengths[task_id] = how many query points for task_id
            lengths = [qX.size(0) for qX in query.values()]
            for k, tensor in data.items():
                if isinstance(tensor, torch.Tensor):
                    structured_data[k] = self.helper_chunk(
                        tensor, query.keys(), lengths)

            self.data[requester] = structured_data['X_neighbors']
            self.extra_info[requester] = structured_data

    def add_data_task_neighbors_prefilter(self, neighbor_id, task_id):
        extra_info = self.incoming_extra_info[neighbor_id]
        task_neighbors_prefilter = extra_info[
            'task_neighbors_prefilter'][task_id]
        valid_rows_mask = ~torch.all(
            task_neighbors_prefilter == -1, dim=1)
        n_neighbor = task_neighbors_prefilter.shape[1]
        valid_mask = valid_rows_mask.unsqueeze(
            1).expand(-1, n_neighbor).reshape(-1)
        return valid_mask

    def add_data_global_y_prefilter(self, neighbor_id, task_id):
        extra_info = self.incoming_extra_info[neighbor_id]
        shared_global_Y = extra_info['Y_neighbors'][task_id]
        shared_local_Y = get_local_labels_for_task(
            shared_global_Y.flatten(), task_id, self.dataset.class_sequence, self.dataset.num_classes_per_task)
        valid_mask = shared_local_Y != -1
        return valid_mask

    def add_data_prefilter(self, neighbor_id, task_id):
        if self.sharing_strategy['add_data_prefilter_strategy'] == 'task_neighbors_prefilter':
            return self.add_data_task_neighbors_prefilter(neighbor_id, task_id)
        elif self.sharing_strategy['add_data_prefilter_strategy'] == 'global_y_prefilter':
            return self.add_data_global_y_prefilter(neighbor_id, task_id)
        elif self.sharing_strategy['add_data_prefilter_strategy'] == 'both':
            return self.add_data_task_neighbors_prefilter(neighbor_id, task_id) & self.add_data_global_y_prefilter(
                neighbor_id, task_id)
        else:
            raise ValueError(
                f"Invalid prefilter strategy {self.sharing_strategy['add_data_prefilter_strategy']}")

    def assign_labels_same_as_query(self, neighbor_id, task_id):
        Y = self.query_y[task_id]  # Y.shape = (N_query)
        n_neighbor = self.incoming_data[neighbor_id][task_id].shape[1]
        Y = Y.unsqueeze(1).expand(-1, n_neighbor).reshape(-1)
        return Y

    def get_query(self):
        return self.query

    def assign_labels_groundtruth(self, neighbor_id, task_id):
        extra_info = self.incoming_extra_info[neighbor_id]
        shared_global_Y = extra_info['Y_neighbors'][task_id]
        shared_local_Y = get_local_labels_for_task(
            shared_global_Y.flatten(), task_id, self.dataset.class_sequence, self.dataset.num_classes_per_task)
        return shared_local_Y

    def assign_labels_to_shared_data(self, neighbor_id, task_id):
        if self.sharing_strategy['assign_labels_strategy'] == 'same_as_query':
            return self.assign_labels_same_as_query(neighbor_id, task_id)
        elif self.sharing_strategy['assign_labels_strategy'] == 'groundtruth':
            return self.assign_labels_groundtruth(neighbor_id, task_id)
        else:
            raise ValueError(
                f"Invalid assign labels strategy {self.sharing_strategy['assign_labels_strategy']}")

    def add_incoming_data(self):
        # get the data and now learn from it
        for neighbor_id, neighbor_data in self.incoming_data.items():
            for task_id, task_data in neighbor_data.items():
                # task_data.shape = (N_query, N_neighbor, C, H, W)
                if task_id not in self.agent.shared_replay_buffers:
                    self.agent.shared_replay_buffers[task_id] = ReplayBufferReservoir(
                        self.sharing_strategy.shared_memory_size, task_id)

                valid_mask = self.add_data_prefilter(neighbor_id, task_id)
                if len(valid_mask) == 0:
                    # logging.info("ERR: No valid neighbor data at task {} for node {} query Y {}".format(
                    #     task_id, self.node_id, self.query_extra_info['query_global_y']))
                    # logging.info("ERR: from {} neighbor y {}".format(neighbor_id,
                    #                                                  self.incoming_extra_info[neighbor_id]['Y_neighbors']))
                    continue

                Y = self.assign_labels_to_shared_data(neighbor_id, task_id)
                Y = Y[valid_mask]

                X = task_data.reshape(
                    -1, *task_data.shape[2:])
                X = X[valid_mask]

                self.agent.shared_replay_buffers[task_id].push(
                    X, Y)

    def get_query_global_labels(self, y):
        ret = {}
        for task, y_t in y.items():
            ret[task] = get_global_labels(
                y_t, [task] * len(y_t), self.dataset.class_sequence, self.dataset.num_classes_per_task)
        return ret

    def prepare_communicate(self, task_id, end_epoch, comm_freq, num_epochs, communication_round, final=False,):
        if communication_round % 2 == 0:
            self.incoming_query, self.incoming_data, self.incoming_extra_info, self.incoming_query_extra_info = {}, {}, {}, {}
        if task_id < self.agent.net.num_init_tasks:
            return
        if communication_round % 2 == 0:
            if 'query_task_mode' not in self.sharing_strategy:
                mode = "all"
                if self.is_modular:
                    component_update_freq = self.train_kwargs['component_update_freq']
                    next_end_epoch = min(end_epoch + comm_freq, num_epochs)
                    has_comp_update = component_update_freq is not None and next_end_epoch % component_update_freq == 0
                    if not has_comp_update:
                        mode = "current"
            else:
                mode = self.sharing_strategy['query_task_mode']
            X, y = self.compute_query(task_id, mode=mode)
            self.query = X
            self.query_y = y
            self.query_extra_info = {
                "query_global_y": self.get_query_global_labels(y),
            }
        elif communication_round % 2 == 1:
            self.compute_data()
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    # potentially parallelizable
    def communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks:
            # NOTE: don't communicate for the first few tasks to
            # allow agents some initital training to find their weakness
            return
        if communication_round % 2 == 0:
            # send query to neighbors
            for neighbor in self.neighbors.values():
                neighbor.receive(self.node_id, self.query, "query")
                neighbor.receive(
                    self.node_id, self.query_extra_info, "query_extra_info")
        elif communication_round % 2 == 1:
            # send data to the requester
            # for requester in self.incoming_query:
            #     self.neighbors[requester].receive(
            #         self.node_id, self.data[requester], "data")
            for neighbor_id, neighbor in self.neighbors.items():
                neighbor.receive(
                    self.node_id, self.data[neighbor_id], "data")
                neighbor.receive(
                    self.node_id, self.extra_info[neighbor_id], "extra_info")
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    def process_communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks:
            return
        if communication_round % 2 == 0:
            pass
        elif communication_round % 2 == 1:
            self.add_incoming_data()
        else:
            raise ValueError("Invalid round number")

    def save_debug_data(self):
        """
        Save data for debugging purposes
        """
        if self.agent.T < self.agent.net.num_init_tasks:
            return

        # pickle query, query_extra_info, incoming_data, incoming_extra_info
        with open(f"{self.save_dir}/task_{self.agent.T-1}/query_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'wb') as f:
            pickle.dump(self.query, f)
        with open(f"{self.save_dir}/task_{self.agent.T-1}/query_extra_info_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'wb') as f:
            pickle.dump(self.query_extra_info, f)
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_data_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'wb') as f:
            pickle.dump(self.incoming_data, f)
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_extra_info_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'wb') as f:
            pickle.dump(self.incoming_extra_info, f)
        # pickle incoming_query
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_query_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'wb') as f:
            pickle.dump(self.incoming_query, f)
        # pickle incoming_query_extra_info
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_query_extra_info_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'wb') as f:
            pickle.dump(self.incoming_query_extra_info, f)

    def load_debug_data(self):
        """
        Load data for debugging purposes
        """
        if self.agent.T < self.agent.net.num_init_tasks:
            return

        # pickle query, query_extra_info, incoming_data, incoming_extra_info
        with open(f"{self.save_dir}/task_{self.agent.T-1}/query_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'rb') as f:
            self.query = pickle.load(f)
        with open(f"{self.save_dir}/task_{self.agent.T-1}/query_extra_info_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'rb') as f:
            self.query_extra_info = pickle.load(f)
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_data_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'rb') as f:
            self.incoming_data = pickle.load(f)
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_extra_info_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'rb') as f:
            self.incoming_extra_info = pickle.load(f)

        # pickle incoming_query
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_query_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'rb') as f:
            self.incoming_query = pickle.load(f)
        # pickle incoming_query_extra_info
        with open(f"{self.save_dir}/task_{self.agent.T-1}/incoming_query_extra_info_{self.sharing_strategy['prefilter_strategy']}_{self.sharing_strategy.scorer}.pt", 'rb') as f:
            self.incoming_query_extra_info = pickle.load(f)

    def get_format_viz_data(self, task):
        anchor_X, anchor_y = self.get_candidate_data(task)
        X_list = []
        y_list = []

        neighbor_ids = self.incoming_query.keys()
        for neighbor_id in neighbor_ids:
            tasks = self.incoming_query[neighbor_id].keys()
            for t in tasks:
                X_list.append(self.incoming_query[neighbor_id][t])
                y_list.append(
                    self.incoming_query_extra_info[neighbor_id]['query_global_y'][t])

        # Concatenate the lists into numpy arrays
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        y_local = get_local_labels_for_task(y, task, self.dataset.class_sequence,
                                            self.dataset.num_classes_per_task)
        # divide X, y into OD and ID. Note that `get_local_labels_for_task`
        # will return -1 for OD data points.
        id_mask = y_local != -1
        od_mask = ~id_mask

        id_X = X[id_mask]
        id_y = y_local[id_mask]

        od_X = X[od_mask]
        od_y = y_local[od_mask]

        data_dict = {
            "ID": (id_X, id_y),
            "anchor": (anchor_X, anchor_y),
            "OD": (od_X, od_y)
        }
        return data_dict


@ray.remote
class ParallelRecvDataAgent(RecvDataAgent):
    def communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks:
            # NOTE: don't communicate for the first few tasks to
            # allow agents some initital training to find their weakness
            return
        if communication_round % 2 == 0:
            # send query to neighbors
            for neighbor in self.neighbors.values():
                ray.get(neighbor.receive.remote(
                    self.node_id, self.query, "query"))
                ray.get(neighbor.receive.remote(
                    self.node_id, self.query_extra_info, "query_extra_info"
                ))
        elif communication_round % 2 == 1:
            # send data to the requester
            for requester in self.incoming_query:
                requester_node = self.neighbors[requester]
                ray.get(requester_node.receive.remote(
                    self.node_id, self.data[requester], "data"))
                ray.get(requester_node.receive.remote(
                    self.node_id, self.extra_info[requester], "extra_info"))
        else:
            raise ValueError(f"Invalid round number {communication_round}")
