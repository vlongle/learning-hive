'''
File: /modmod.py
Project: fleet
Created Date: Tuesday March 28th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


from shell.fleet.fleet import Agent
from shell.fleet.data.data_utilize import compute_tasks_sim
import ray
from shell.utils.record import Record
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from shell.fleet.data.recv import random_scorer, cross_entropy_scorer, compute_embedding_dist


def create_general_permutation_matrix(task1_classes, task2_classes):
    # Initialize a zero matrix for the new definition
    assert len(task1_classes) == len(task2_classes)
    K = len(task2_classes)
    P = torch.zeros(K, K)

    task1_mapping = {c: i for i, c in enumerate(task1_classes)}
    task2_mapping = {c: i for i, c in enumerate(task2_classes)}

    for class_ in task1_classes:
        if class_ in task2_mapping:
            P[task2_mapping[class_], task1_mapping[class_]] = 1

    return P


# Function to apply transformation to D and create D'
def transform_D(D, P):
    F, K = D.weight.data.shape[1], D.weight.data.shape[0]

    # transformed_weights = torch.matmul(P, D.weight.data)
    # transformed_bias = torch.matmul(P, D.bias.data)
    transformed_weights = torch.matmul(P, D.weight.data)
    transformed_bias = torch.matmul(P, D.bias.data)
    D_prime = nn.Linear(F, K, bias=True)
    D_prime.weight.data = transformed_weights
    D_prime.bias.data = transformed_bias
    return D_prime


class ModuleRanker:
    def __init__(self, agent):
        self.agent = agent

    def compute_task_sims(self, neighbor_id, task_id):
        raise NotImplementedError

    def send_most_similar_module(self, neighbor_id, task_id):
        task_sims = self.compute_task_sims(neighbor_id, task_id)
        # print('task_sims', self.agent.node_id,
        #       neighbor_id, task_id, '>', task_sims)
        module_record = self.agent.agent.dynamic_record.df

        for t in range(len(task_sims)):
            if t not in set(module_record['task_id']):
                task_sims[t] = float('-inf')
            if t in set(module_record['task_id']) and not module_record[module_record['task_id'] == t]['add_new_module'].item():
                task_sims[t] = float('-inf')

        # print('AFTER PROCESSED: task_sims', self.agent.node_id,
        #       neighbor_id, task_id, '>', task_sims)
        # get the most similar task with the highest similarity. Break ties by the task id
        # (highest wins)
        # most_similar_task = max(
        #     range(len(task_sims)), key=lambda x: (task_sims[x], x))
        # lowest task_id wins
        most_similar_task = max(
            range(len(task_sims)), key=lambda x: (task_sims[x], -x))
        if task_sims[most_similar_task] == float('-inf'):
            return []

        task_module = module_record[module_record['task_id']
                                    == most_similar_task]['num_components'].item() - 1
        # print('node', self.node_id, 'for neighbor', neighbor_id, '@ task', task_id, 'sending module', task_module, 'from task', most_similar_task,
        #       'with similarity', task_sims[most_similar_task], 'current no. of modules', len(self.net.components))
        # pathological for replaying ipynb
        if task_module >= len(self.agent.net.components):
            return []
        return [{'source_task_id': most_similar_task,
                 'task_sim': task_sims[most_similar_task],
                 'module_id': task_module,
                 'module': self.agent.net.components[task_module],
                 'decoder': self.agent.net.decoder[most_similar_task],
                 'structure': self.agent.net.structure[most_similar_task],
                 'source_class_labels': self.agent.dataset.class_sequence[most_similar_task * self.agent.dataset.num_classes_per_task:
                                                                          (most_similar_task + 1) * self.agent.dataset.num_classes_per_task]},]

    def select_module(self, neighbor_id, task_id):
        outgoing_modules = {}
        if self.agent.sharing_strategy.module_selection == "naive":
            # send all the new modules
            num_newly_added_modules = len(
                self.agent.net.components) - self.agent.net.depth - len(self.agent.net.candidate_indices)
            outgoing_modules = []
            for i in range(num_newly_added_modules):
                assert i not in self.agent.net.candidate_indices
                outgoing_modules.append(self.agent.net.components[i])
        elif self.agent.sharing_strategy.module_selection == "gt_most_similar":
            outgoing_modules = self.send_most_similar_module(
                neighbor_id, task_id)
        else:
            raise NotImplementedError(
                f"Module selection {self.agent.sharing_strategy.module_selection} not implemented.")

        return outgoing_modules


class GlobalLabelsModuleRanker(ModuleRanker):
    def compute_task_similarity(self, neighbor_task, task_id):
        candidate_tasks = [self.agent.dataset.class_sequence[t *
                                                             self.agent.dataset.num_classes_per_task: (t+1) * self.agent.dataset.num_classes_per_task] for t in range(task_id + 1)]
        return [compute_tasks_sim(task, neighbor_task) for task in candidate_tasks]

    def compute_task_sims(self, neighbor_id, task_id):
        neighbor_task = self.agent.query_tasks[neighbor_id]
        task_sims = self.compute_task_similarity(
            neighbor_task, task_id)
        return task_sims

    def send_query(self, task_id):
        task = self.agent.dataset.class_sequence[task_id *
                                                 self.agent.dataset.num_classes_per_task: (task_id + 1) * self.agent.dataset.num_classes_per_task]
        return task


# LEEP
def log_expected_empirical_prediction(predictions: np.ndarray, labels: np.ndarray):
    r"""
    Log Expected Empirical Prediction in `LEEP: A New Measure to
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>`_.

    The LEEP :math:`\mathcal{T}` can be described as:

    .. math::
        \mathcal{T}=\mathbb{E}\log \left(\sum_{z \in \mathcal{C}_s} \hat{P}\left(y \mid z\right) \theta\left(y \right)_{z}\right)

    where :math:`\theta\left(y\right)_{z}` is the predictions of pre-trained model on source category, :math:`\hat{P}\left(y \mid z\right)` is the empirical conditional distribution estimated by prediction and ground-truth label.

    Args:
        predictions (np.ndarray): predictions of pre-trained model.
        labels (np.ndarray): groud-truth labels.

    Shape: 
        - predictions: (N, :math:`C_s`), with number of samples N and source class number :math:`C_s`.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar
    """
    N, C_s = predictions.shape
    labels = labels.reshape(-1)
    # print('labels', labels)
    C_t = int(np.max(labels) + 1)

    normalized_prob = predictions / float(N)
    # placeholder for joint distribution over (y, z)
    joint = np.zeros((C_t, C_s), dtype=float)

    for i in range(C_t):
        this_class = normalized_prob[labels == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    p_target_given_source = (
        joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    empirical_prediction = predictions @ p_target_given_source
    empirical_prob = np.array(
        [predict[label] for predict, label in zip(empirical_prediction, labels)])
    score = np.mean(np.log(empirical_prob))

    return score


@torch.inference_mode()
def predict(net, X, task_id, get_probs=False):
    was_training = net.training
    net.eval()
    # Forward pass: Get predictions for X
    predictions = net(X, task_id)
    # print('prediction', predictions)

    # Assuming the network outputs logits, apply softmax to get probabilities
    probabilities = torch.softmax(predictions, dim=1)
    if get_probs:
        return probabilities

    # Convert probabilities to predicted labels
    _, predicted_labels = torch.max(probabilities, dim=1)

    # Restore original training state of the network
    if was_training:
        net.train()

    return predicted_labels


@torch.inference_mode()
def compute_consistency(net, X, task_id):
    predicted_labels = predict(net, X, task_id)
    # Find the most common label
    most_common_label = predicted_labels.mode().values.item()
    # print('most_common_label', most_common_label)

    # Compute inconsistency as the fraction of labels that are not the most common one
    consistency = (predicted_labels == most_common_label).float().mean()

    return consistency.item()


class InstanceMapModuleRanker(ModuleRanker):
    def __init__(self, agent):
        super().__init__(agent)
        self.scorer = random_scorer
        self.scorer_type = 'unsupervised'
        # self.scorer = cross_entropy_scorer
        # self.scorer_type = 'supervised'
        self.dist = {}

    @torch.inference_mode()
    def send_query(self, task_id, k=40):
        was_training = self.agent.net.training
        self.agent.net.eval()
        X_val, Y_val = self.agent.dataset.valset[task_id].tensors
        res = {}
        for y in Y_val.unique():
            X_val_y = X_val[Y_val == y].to(self.agent.net.device)
            y_val_y = Y_val[Y_val == y].to(self.agent.net.device)
            logits = self.agent.net(X_val_y, task_id)
            scores = self.scorer(
                logits) if self.scorer_type == "unsupervised" else self.scorer(logits, y_val_y)
            scores = -1.0 * scores
            # print('y', y, 'X_val_y', X_val_y.shape, 'max score',
            #       scores.max(), 'min score', scores.min())
            # pick the top k scores
            topk = torch.topk(scores, min(k, len(scores))).indices
            res[y.item()] = {'X': X_val_y[topk],
                             'scores': scores[topk]}

        if was_training:
            self.agent.net.train()
        return res

    # @torch.inference_mode()
    # def compute_task_similarity(self, neighbor_id, task_id, k=10):
    #     was_training = self.agent.net.training
    #     self.agent.net.eval()
    #     query = self.agent.query_tasks[neighbor_id]
    #     X_val, Y_val = self.agent.dataset.valset[task_id].tensors
    #     res = {}
    #     baseline = {}  # compute avg distance of cluster yp
    #     for yp in Y_val.unique():
    #         X_val_yp = X_val[Y_val == yp].to(self.agent.net.device)
    #         dist = compute_embedding_dist(
    #             self.agent.net, X_val_yp, task_id=task_id)  # this is actually similarity NOT distance
    #         dist, _ = torch.topk(dist, k=1)
    #         baseline[yp.item()] = dist.mean().item()
    #         print('baseline', yp, baseline[yp.item()])

    #     for y, data in query.items():
    #         for yp in Y_val.unique():
    #             X_val_yp = X_val[Y_val == yp].to(self.agent.net.device)
    #             # dist = (num_query, N) where N is len(X_val_yp) and num_query is len(data['X']
    #             dist = compute_embedding_dist(
    #                 self.agent.net, data['X'], X_val_yp, task_id)

    #             # reduce to (num_query, k) by taking the smallest k distances
    #             dist, _ = torch.topk(dist, k)
    #             print('>> dist:', dist.shape)
    #             # dist = dist.mean(dim=1)
    #             dist = dist.mean().item() - baseline[yp.item()]
    #             # # turn dist to 0 or 1 based on the baseline
    #             # dist = dist > baseline[yp.item()]
    #             res[(y, yp.item())] = dist
        # self.dist[(neighbor_id, task_id)] = res
        # if was_training:
        #     self.agent.net.train()
        # return res

    @torch.inference_mode()
    def compute_task_similarity(self, neighbor_id, task_id):
        query = self.agent.query_tasks[neighbor_id]
        # const = 0
        # for y, data in query.items():
        #     X = data['X'].to(self.agent.net.device)
        #     c = compute_consistency(self.agent.net, X, task_id)
        #     print('neighbor', neighbor_id, 'y', y,
        #           'task_id', task_id, 'consistency', c)
        #     const += c
        # return const

        y_preds, ys = [], []
        for y, data in query.items():
            pred = predict(self.agent.net, data['X'], task_id, get_probs=True)
            N, C = pred.shape
            targets = torch.full((N,), y)
            y_preds.append(pred)
            ys.append(targets)

        y_preds = torch.cat(y_preds)
        ys = torch.cat(ys).flatten()
        # print('y', ys.shape, 'y_preds.shape', y_preds.shape)
        return log_expected_empirical_prediction(y_preds.cpu().numpy(), ys.cpu().numpy())

    def compute_task_sims(self, neighbor_id, task_id):
        return [self.compute_task_similarity(neighbor_id, t)for t in range(task_id)]


class ModuleSelection:
    pass


class TrustSimModuleSelection(ModuleSelection):
    def choose_best_module_from_neighbors(self, module_list):
        # module_list is a list of (t, sim, module)
        # find the most similar task based on the similarity score. Break ties by the task id
        # (highest task id wins)
        # best_match = max(module_list, key=lambda x: (
        #     x[1], x[0]))
        # lowest task_id wins
        best_match_index = max(enumerate(module_list),
                               key=lambda x: (x[1]['task_sim'], -x[1]['source_task_id']))[0]
        return best_match_index


class TryOutModuleSelection(ModuleSelection):
    def choose_best_module_from_neighbors(self, module_list):
        pass


class ModModAgent(Agent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.modmod_record = Record(
            f"{self.save_dir}/modmod_add_modules_record.csv")

        if self.sharing_strategy.ranker == 'label':
            self.module_ranker = GlobalLabelsModuleRanker(self)
        elif self.sharing_strategy.ranker == 'instance':
            self.module_ranker = InstanceMapModuleRanker(self)
        else:
            raise NotImplementedError(
                f"Ranker {self.sharing_strategy.ranker} not implemented.")

        if self.sharing_strategy.module_select == "trust_sim":
            self.module_select = TrustSimModuleSelection()
        elif self.sharing_strategy.module_select == "tryout":
            self.module_select = TryOutModuleSelection()
        else:
            raise NotImplementedError(
                f"Module selection {self.sharing_strategy.module_select} not implemented.")

    def load_records(self):
        self.modmod_record.df = pd.read_csv(
            f"{self.save_dir}/modmod_add_modules_record.csv")
        return super().load_records()

    def change_save_dir(self, save_dir):
        self.save_dir = save_dir
        self.modmod_record.path = f"{self.save_dir}/modmod_add_modules_record.csv"
        return super().change_save_dir(save_dir)

    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True, **kwargs):

        module_list = self.train_kwargs.get("module_list", [])
        decoder_list = self.train_kwargs.get("decoder_list", [])
        structure_list = self.train_kwargs.get("structure_list", [])
        if self.sharing_strategy.opt_with_random:
            # optimize the received module along with a random module
            # as well
            num_candidate_modules = len(module_list) + 1
        else:
            num_candidate_modules = len(module_list)

        if len(module_list) == 0:
            num_candidate_modules = 1  # at the very least, we need to consider a random module

        if "num_candidate_modules" not in kwargs:  # HACK: to be compatible with the exploratory ipynb
            kwargs["num_candidate_modules"] = num_candidate_modules

        train_candidate_module = not self.sharing_strategy.freeze_candidate_module

        # print(f"BEFORE TRAINING {self.node_id} module_list {len(module_list)} decoder_list _{len(decoder_list)} structure_list {len(structure_list)}")
        if self.sharing_strategy.transfer_decoder and len(decoder_list) > 0:
            self.transfer_decoder(
                task_id, decoder_list[0])
        if self.sharing_strategy.transfer_structure and len(structure_list) > 0:
            self.transfer_structure(
                task_id, structure_list[0])
        if "decoder_list" in self.train_kwargs:
            del self.train_kwargs["decoder_list"]
        if "structure_list" in self.train_kwargs:
            del self.train_kwargs["structure_list"]

        return super().train(task_id, start_epoch, communication_frequency, final, train_candidate_module=train_candidate_module,
                             **kwargs)

    def transfer_decoder(self, task_id, decoder):
        decoder, source_class_labels = decoder['decoder'], decoder['source_class_labels']

        P = create_general_permutation_matrix(
            source_class_labels, self.dataset.class_sequence[task_id * self.dataset.num_classes_per_task:
                                                             (task_id + 1) * self.dataset.num_classes_per_task],)
        new_decoder = transform_D(decoder, P.to(self.net.device))
        self.net.decoder[task_id].load_state_dict(new_decoder.state_dict())

    def transfer_structure(self, task_id, structure, value=None):
        # print("before structure transfer", self.node_id, self.net.structure[task_id].shape, 'no_comp', self.net.num_components, 'len(comp)', len(self.net.components))
        if value is None:
            # value = -np.inf
            value = 0.0
        new_s = structure['structure'][:self.net.num_init_tasks, :].data
        new_s = torch.cat((new_s, torch.full((len(self.net.components)-self.net.num_init_tasks, self.net.depth), value,
                                             device=self.net.device)),
                          dim=0)
        shared_module_weight = structure['structure'][structure['module_id']].data
        new_s = torch.cat((new_s, shared_module_weight.view(1, -1)), dim=0)
        self.net.structure[task_id].data = new_s
        # print("after structure transfer", self.node_id,
        #       self.net.structure[task_id].shape)

    def prepare_communicate(self,  task_id, end_epoch, comm_freq, num_epochs, communication_round,
                            final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 1:
            self.outgoing_modules = {}
            self.incoming_modules = {}
            for neighbor_id, neighbor in self.neighbors.items():
                self.outgoing_modules[neighbor_id] = self.module_ranker.select_module(
                    neighbor_id, task_id)
        else:
            self.query_tasks = {}

    def receive(self, sender_id, data, msg_type):
        if msg_type == "query_task":
            self.query_tasks[sender_id] = data
        elif msg_type == "module":
            self.incoming_modules[sender_id] = data

    def communicate(self, task_id, communication_round, final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 0:
            self.send_query_task(task_id)
        else:
            for neighbor_id, neighbor in self.neighbors.items():
                neighbor.receive(
                    self.node_id, self.outgoing_modules[neighbor_id], "module")

    def get_module_list(self):
        module_list = []
        for neighbor_id in self.neighbors:
            extra_info_ls = [e | {'neighbor_id': neighbor_id}
                             for e in self.incoming_modules[neighbor_id]]
            module_list += extra_info_ls
        return module_list

    def process_communicate(self, task_id, communication_round, final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 1:
            module_list = self.get_module_list()
            if len(module_list) == 0:
                self.train_kwargs["module_list"] = []
                self.train_kwargs["decoder_list"] = []
                self.train_kwargs["structure_list"] = []
                row = {
                    'task_id': task_id,
                    "source_task_id": -1,
                    'task_sim': 0,
                    'module_id': -1,
                    # 'source_class_labels': None,
                    'neighbor_id': -1,
                }
            else:
                best_match = module_list[self.module_select.choose_best_module_from_neighbors(
                    module_list)]
                self.train_kwargs["module_list"] = [best_match['module']]
                self.train_kwargs["decoder_list"] = [{"decoder": best_match['decoder'],
                                                      "source_class_labels": best_match['source_class_labels']}]
                self.train_kwargs["structure_list"] = [{'structure': best_match['structure'],
                                                        'module_id': best_match['module_id']}]
                # Create a new dictionary with task_id as the first key
                row = {"task_id": task_id}
                # Update the new dictionary with the keys and values from best_match
                row.update(best_match)
                del row['module']
                del row['decoder']
                del row['structure']
                del row['source_class_labels']

            # print('row', row)

            # record for the modmod record
            self.modmod_record.write(
                row
            )
            self.modmod_record.save()
        # else:
        #     self.task_sims = {}
        #     for neighbor_id in self.neighbors:
        #         neighbor_task = self.query_tasks[neighbor_id]
        #         self.task_sims[neighbor_id] = self.compute_task_similarity(
        #             neighbor_task, task_id)

    def send_query_task(self, task_id):
        query = self.module_ranker.send_query(task_id)
        for neighbor in self.neighbors.values():
            neighbor.receive(self.node_id, query, "query_task")


@ray.remote
class ParallelModModAgent(ModModAgent):
    def send_query_task(self, task_id):
        query = self.module_ranker.send_query(task_id)

        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(self.node_id, query, "query_task"))

    def communicate(self, task_id, communication_round, final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 0:
            self.send_query_task(task_id)
        else:
            for neighbor_id, neighbor in self.neighbors.items():
                ray.get(neighbor.receive.remote(
                    self.node_id, self.outgoing_modules[neighbor_id], "module"))
