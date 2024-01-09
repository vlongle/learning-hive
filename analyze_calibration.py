import omegaconf
from shell.utils.experiment_utils import *
from shell.fleet.utils.fleet_utils import *
from shell.utils.metric import *
import matplotlib.pyplot as plt
from shell.fleet.network import TopologyGenerator
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from shell.fleet.fleet import Agent, Fleet
from shell.fleet.data.data_utilize import *
from shell.fleet.data.recv import *

from sklearn.manifold import TSNE
from torchvision.utils import make_grid
from shell.fleet.data.data_utilize import *
import logging
from sklearn.metrics import f1_score
import os
from shell.fleet.data.recv_utils import *
from tqdm import tqdm
import argparse
from functools import partial
from torchvision.utils import make_grid
from shell.utils.oodloss import OODSeparationLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from shell.utils.record import Record


@torch.inference_mode()
def contrastive_transform(net, task, X):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.to(net.device)
    return net.contrastive_embedding(X, task).detach().cpu().numpy()


def apply_transform_to_data_dict(data_dict, transform=None):
    """
    Applies a transformation function to the X component of each dataset in the data_dict.

    :param data_dict: Dictionary containing datasets. Each key corresponds to a dataset name,
                      and each value is a tuple (X, y) where X is the data to transform.
    :param transform: A function that applies a transformation to X.
    :return: A new data_dict with transformed X components.
    """
    transformed_data_dict = {}
    if transform is None:
        def transform(x): return x

    for key, (X, y) in data_dict.items():
        # Apply the transform function to X
        transformed_X = transform(X)

        # Update the dataset in the new dictionary
        transformed_data_dict[key] = (transformed_X, y)

    return transformed_data_dict


def calculate_boxplot_outlier_thresholds(data):
    """
    Calculate the lower and upper threshold for outliers based on the boxplot method.

    Parameters:
    data (array-like): The input data to calculate the thresholds.

    Returns:
    (float, float): A tuple containing the lower and upper thresholds for outliers.
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    return lower_threshold, upper_threshold


@torch.inference_mode()
def knn_distance(net, task_id, anchor_X, X, num_neighbors=1, exclude_self=True):
    anchor_embed = contrastive_transform(net, task_id, anchor_X)
    X_embed = contrastive_transform(net, task_id, X)

    # Compute pairwise distances using broadcasting
    distances = np.linalg.norm(X_embed[:, np.newaxis] - anchor_embed, axis=2)

    if exclude_self:
        # Create a mask where each element compares X_embed to each anchor_embed
        # The mask is True where embeddings are equal
        mask = np.all(np.isclose(X_embed[:, np.newaxis], anchor_embed), axis=2)

        # Set distances to infinity where the mask is True
        distances[mask] = np.inf

    # Sort distances for each element in X_embed and take the average of the nearest 'num_neighbors'
    sorted_distances = np.sort(distances, axis=1)
    avg_nearest_distances = np.mean(
        sorted_distances[:, :num_neighbors], axis=1)

    return avg_nearest_distances


dataset = "mnist"
algo = "modular"
prefilter_strategy = "None"
scorer = "cross_entropy"

experiment_folder = "experiment_results"
# experiment_name = "vanilla_fix_bug_compute_loss_encodev2"
# experiment_name = "vanilla_ood_separation_loss"
experiment_name = "vanilla_fix_bug_compute_loss_encodev2"
# experiment_name = "test"

use_contrastive = True
num_trains_per_class = 64
seed = 0
num_tasks = 10
parallel = False
comm_freq = None  # "None" means no communication, doesn't matter for this analysis
num_tasks = 10

if __name__ == "__main__":

    save_dir = get_save_dir(experiment_folder, experiment_name,
                            dataset, algo, num_trains_per_class, use_contrastive, seed)
    graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg, cfg = get_cfg(
        save_dir)

    cfg.sharing_strategy = DictConfig({
        "name": "recv_data",
        "scorer": scorer,
        "num_queries": 5,
        'num_data_neighbors': 5,
        'num_filter_neighbors': 5,
        'num_coms_per_round': 2,
        "query_score_threshold": 0.0,
        "shared_memory_size": 50,
        "comm_freq": comm_freq,
        "prefilter_strategy": prefilter_strategy,
        "use_ood_separation_loss": True,
    })
    AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.algo, parallel)
    FleetCls = get_fleet(cfg.sharing_strategy, parallel)
    fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                     LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                     train_kwargs=train_cfg, **fleet_additional_cfg)

    fleet.load_model_from_ckpoint(task_ids=num_tasks-1)
    fleet.update_replay_buffers(num_tasks-1)
    # print(fleet.eval_test())

    record = Record(f"analyze_calibration_{experiment_name}.csv")

    for agent_id, agent in enumerate(fleet.agents):
        for t in range(num_tasks):
            anchor_X, anchor_y, _ = agent.agent.replay_buffers[t].tensors
            # anchor_X, anchor_y = agent.dataset.trainset[t].tensors
            X_od, y_od, X_id, y_id = agent.get_ood_data(t)
            data_dict = {
                "anchor": (anchor_X, anchor_y),
                # "OD": (X_od, y_od),
                # "ID": (X_id, y_id),
            }
            if X_od is None or len(X_od) > 0:
                data_dict["OD"] = (X_od, y_od)
            if X_id is None or len(X_id) > 0:
                data_dict["ID"] = (X_id, y_id)

            computer = partial(knn_distance, agent.net, t,
                               anchor_X, num_neighbors=5)
            # Compute KNN distances for each group
            scores_dict = {}
            for key in ['anchor', 'ID', 'OD']:
                if key in data_dict:
                    X, _ = data_dict[key]
                    scores = computer(X)
                    scores_dict[key] = scores
            if 'OD' not in scores_dict:
                scores_dict['OD'] = np.array([])
            if 'ID' not in scores_dict:
                scores_dict['ID'] = np.array([])

            all_distances = scores_dict['anchor'].tolist(
            ) + scores_dict['ID'].tolist() + scores_dict['OD'].tolist()
            _, upper_threshold = calculate_boxplot_outlier_thresholds(
                scores_dict['anchor'])
            labels = all_distances > upper_threshold
            # ground_truth_combined = [0] * len(scores_dict['ID']) + [1] * len(scores_dict['OD'])
            ground_truth_combined = [
                0] * len(scores_dict['ID']) + [1] * len(scores_dict['OD'])

            # Since labels include 'anchor', we need to exclude them from accuracy calculation
            # Assuming the length of 'anchor' is known
            anchor_length = len(scores_dict['anchor'])
            id_od_labels = labels[anchor_length:]  # Excluding anchor labels

            # Calculate various metrics using sklearn
            accuracy = accuracy_score(ground_truth_combined, id_od_labels)
            precision = precision_score(ground_truth_combined, id_od_labels)
            recall = recall_score(ground_truth_combined, id_od_labels)
            f1 = f1_score(ground_truth_combined, id_od_labels)
            record.write({
                'agent_id': agent_id,
                'task_id': t,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })

    print(record.df)
    print(record.df.groupby('task_id').mean())
    print(record.df.groupby('agent_id').mean())
    # average over all agent and all tasks
    print(record.df.groupby('task_id').mean().mean())
    record.save()
