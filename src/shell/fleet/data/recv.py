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


class OODSeparationLoss(torch.nn.Module):
    def __init__(self, delta=1.0, lambda_ood=1.0):
        """
        OOD Separation Loss to push away OOD data from task-specific data.
        :param delta: Margin threshold for the OOD separation.
        :param lambda_ood: Weighting factor for the OOD loss.
        """
        super().__init__()
        self.delta = delta
        self.lambda_ood = lambda_ood

    def forward(self, task_embeddings, ood_embeddings):
        """
        Compute the OOD separation loss.
        :param task_embeddings: Embeddings of the current task data.
        :param ood_embeddings: Embeddings of the OOD data.
        :return: OOD separation loss.
        """
        # Compute pairwise distance matrix between task and OOD embeddings
        dist_matrix = torch.cdist(task_embeddings, ood_embeddings, p=2)

        # Apply margin threshold
        margin_violations = torch.relu(self.delta - dist_matrix)

        # Compute mean of the violations
        ood_loss = margin_violations.mean()

        return self.lambda_ood * ood_loss


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
        X1_embed = self.net.encode(
            X1.to(self.net.device), task_id=task_id)  # (B, hidden_dim)
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

    def get_ood_data(self, task_id):
        # Get the class labels for the current task
        task_classes = list(self.dataset.class_sequence[task_id * self.dataset.num_classes_per_task:
                                                        (task_id + 1) * self.dataset.num_classes_per_task])

        # Gather data from replay buffers of all tasks except the current task
        replay_buffers = {t: self.agent.replay_buffers[t] for t in range(self.agent.T)
                          if t != task_id}

        X_ood = torch.cat([rb.tensors[0]
                           for t, rb in replay_buffers.items()], dim=0)
        y_ood = torch.cat([torch.from_numpy(get_global_label(rb.tensors[1],
                                                             t, self.dataset.class_sequence,
                                                             self.dataset.num_classes_per_task))
                           for t, rb in replay_buffers.items()], dim=0)

        # Convert task_classes to a tensor for efficient comparison
        task_classes_tensor = torch.tensor(task_classes)

        # Find indices of samples in y_ood that do not belong to the current task's classes
        mask = ~y_ood.unsqueeze(1).eq(task_classes_tensor).any(1)
        X_ood_filtered = X_ood[mask]
        y_ood_filtered = y_ood[mask]

        X_iid_filtered = X_ood[~mask]
        y_iid_filtered = y_ood[~mask]

        return X_ood_filtered, y_ood_filtered, X_iid_filtered, y_iid_filtered

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
            computer = self.compute_embedding_dist

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

    def prefilter(self, qX, neighbor_id, n_filter_neighbors):
        if self.sharing_strategy['prefilter_strategy'] == 'raw_distance':
            return self.prefilter_raw_distance(qX, n_filter_neighbors)
        elif self.sharing_strategy['prefilter_strategy'] == 'None':
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

    def prefilter_oracle_helper(self, qX, q_global_Y, n_filter_neighbors):
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

    def prefilter_raw_distance(self, qX, n_filter_neighbors):
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
            qX, computer=self.compute_embedding_dist)

        # 3. Extract top neighbors considering the pre-filtered tasks
        X_neighbors, Y_neighbors, task_neighbors = self.extract_topk_from_similarity(
            sims, Xs, ys, tasks,
            neighbors=n_neighbors,
            # candidate_tasks=task_lists,
            candidate_tasks=prefilter_info['task_neighbors_prefilter'],
        )

        if debug:
            return {
                "X_neighbors": X_neighbors,
                "Y_neighbors": Y_neighbors,
                "task_neighbors": task_neighbors,
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

    def add_incoming_data(self):
        # get the data and now learn from it
        for neighbor_id, neighbor_data in self.incoming_data.items():
            for task_id, task_data in neighbor_data.items():
                # task_data.shape = (N, n_neighbor, C, H, W)
                if task_id not in self.agent.shared_replay_buffers:
                    self.agent.shared_replay_buffers[task_id] = ReplayBufferReservoir(
                        self.sharing_strategy.shared_memory_size, task_id)
                Y = self.query_y[task_id]  # Y.shape = (N)
                # Extracting n_neighbor from task_data shape
                n_neighbor = task_data.shape[1]

                # Expanding Y to shape (N, n_neighbor)
                # and then flattening it to (N*n_neighbor,)
                Y_expanded = Y.unsqueeze(1).expand(-1, n_neighbor).reshape(-1)

                # Flattening task_data to (N*n_neighbor, C, H, W)
                X_flattened = task_data.reshape(-1, *task_data.shape[2:])

                # Storing flattened X and Y into the replay buffer
                self.agent.shared_replay_buffers[task_id].push(
                    X_flattened, Y_expanded)

        # TODO: start learning now!

    def get_query_global_labels(self, y):
        ret = {}
        for task, y_t in y.items():
            ret[task] = get_global_labels(
                y_t, [task] * len(y_t), self.dataset.class_sequence, self.dataset.num_classes_per_task)
        return ret

    def prepare_communicate(self, task_id, communication_round, final=False):
        if communication_round % 2 == 0:
            self.incoming_query, self.incoming_data, self.incoming_extra_info, self.incoming_query_extra_info = {}, {}, {}, {}
        if task_id < self.agent.net.num_init_tasks - 1:
            return
        if communication_round % 2 == 0:
            X, y = self.compute_query(task_id)
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
        if task_id < self.agent.net.num_init_tasks - 1:
            # NOTE: don't communicate for the first few tasks to
            # allow agents some initital training to find their weakness
            return
        if communication_round % 2 == 0:
            # send query to neighbors
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.query, "query")
                neighbor.receive(
                    self.node_id, self.query_extra_info, "query_extra_info")
        elif communication_round % 2 == 1:
            # send data to the requester
            # for requester in self.incoming_query:
            #     self.neighbors[requester].receive(
            #         self.node_id, self.data[requester], "data")
            for neighbor in self.neighbors:
                neighbor.receive(
                    self.node_id, self.data[neighbor.node_id], "data")
                neighbor.receive(
                    self.node_id, self.extra_info[neighbor.node_id], "extra_info")
        else:
            raise ValueError(f"Invalid round number {communication_round}")

    def process_communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks - 1:
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


class ParallelRecvDataAgent(RecvDataAgent):
    def communicate(self, task_id, communication_round, final=False):
        if communication_round % 2 == 0:
            # send query to neighbors
            for neighbor in self.neighbors:
                neighbor.remote.receive(self.node_id, self.query, "query")
        elif communication_round % 2 == 1:
            # send data to the requester
            for requester in self.incoming_query:
                requester.remote.receive(
                    self.node_id, self.data[requester], "data")
        else:
            raise ValueError(f"Invalid round number {communication_round}")
