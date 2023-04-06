'''
File: /recv_test.py
Project: notebook
Created Date: Monday April 3rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
# %% [markdown]
# Comparing different Methods for hard-example mining in `Recv` method
# %%
from shell.fleet.data.recv import least_confidence_scorer, entropy_scorer, margin_scorer, cross_entropy_scorer
import seaborn as sns
import logging
from sklearn.manifold import TSNE
from shell.utils.experiment_utils import setup_experiment
from shell.utils.experiment_utils import eval_net
from shell.learners.er_nocomponents import NoComponentsER
from shell.learners.er_dynamic import CompositionalDynamicER
from shell.models.mlp_soft_lifelong_dynamic import MLPSoftLLDynamic
from shell.models.mlp import MLP
from shell.models.cnn import CNN
from shell.models.cnn_soft_lifelong_dynamic import CNNSoftLLDynamic
from shell.fleet.network import TopologyGenerator
from pprint import pprint
from shell.utils.utils import seed_everything
from omegaconf import DictConfig
import torch.nn as nn
import subprocess
import torch
from shell.utils.utils import seed_everything, viz_embedding
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from shell.datasets.datasets import get_dataset


# %%
logging.basicConfig(level=logging.INFO)

# %%


def train(dataset_name, seed):
    seed_everything(seed)

    if dataset_name == "cifar100":
        # contrastive is taking too long to train
        # for this simple ablation
        use_contrastive = False
        data_cfg = {
            "dataset_name": dataset_name,
            "num_tasks": 4,
            "num_train_per_task": 256,
            "num_classes_per_task": 5,
            "num_val_per_task": 102,
            'remap_labels': True,
            'use_contrastive': use_contrastive,
        }

    else:
        use_contrastive = True
        data_cfg = {
            "dataset_name": dataset_name,
            "num_tasks": 1,
            "num_train_per_task": 128,
            "num_val_per_task": 102,
            'remap_labels': True,
            'use_contrastive': use_contrastive,
        }
    dataset = get_dataset(**data_cfg)

    if dataset_name == "cifar100":
        net_cfg = {
            'depth': 4,
            'num_init_tasks': 4,
            'num_classes': 5,
            'num_tasks': 1,
            "channels": 50,
            "conv_kernel": 3,
            "maxpool_kernel": 2,
            "padding": 1,
            "i_size": 32,
            'dropout': 0.0,
        }
        net = CNN(**net_cfg)

    else:
        net_cfg = {
            'depth': 2,
            'layer_size': 64,
            'num_init_tasks': -1,
            'i_size': 28,
            'num_classes': 2,
            'num_tasks': 1,
            'dropout': 0.0,
        }
        net = MLP(**net_cfg)

    agent_cfg = {
        'memory_size': 64,
        'use_contrastive': use_contrastive,
    }
    agent = NoComponentsER(net, **agent_cfg)

    num_epochs = 500

    for task_id in range(4):
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=128,
                                                         shuffle=False,
                                                         num_workers=0,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}

        trainloader = torch.utils.data.DataLoader(dataset.trainset[task_id],
                                                  batch_size=32,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  )
        agent.train(trainloader, task_id=task_id, num_epochs=num_epochs, testloaders=testloaders,
                    save_freq=1)
    return agent, dataset

# %%


def calculate_scores(agent, valset):
    scorer = {
        'least_confidence': least_confidence_scorer,
        'entropy': entropy_scorer,
        'margin': margin_scorer,
        'cross_entropy': cross_entropy_scorer,
    }

    X_val, Y_val = valset.tensors

    with torch.inference_mode():
        X_val = X_val.to(agent.net.device)
        Y_val = Y_val.to(agent.net.device)
        logits = agent.net(X_val, task_id=0)
        Y_hat = logits.argmax(dim=1)
        acc = (Y_hat == Y_val).float().mean()
        print(acc)
    scores = {}
    for name, score_fn in scorer.items():
        scores[name] = score_fn(logits, Y_val)

    return scores

# %%


def calculate_agreement(scores):
    top_k = 10

    agreements = np.zeros((len(scores), len(scores)))
    # compute the agreement rate between pairs of scorers
    for name1, score1 in scores.items():
        for name2, score2 in scores.items():
            # if name1 == name2:
            #     continue
            # agreement is defined as the size of the intersection of the top k indices
            top_k_idx1 = score1.topk(top_k)[1]
            top_k_idx2 = score2.topk(top_k)[1]
            agreement = len(set(top_k_idx1.tolist()) &
                            set(top_k_idx2.tolist())) / top_k
            agreements[list(scores.keys()).index(name1), list(
                scores.keys()).index(name2)] = agreement
    return agreements


# %%
num_seeds = 4
mean_agreements = np.zeros((num_seeds, 4, 4))
for seed in range(num_seeds):
    agent, dataset = train('cifar100', seed)
    scores = calculate_scores(agent, dataset.valset[0])
    agreements = calculate_agreement(scores)
    mean_agreements[seed] = agreements

# average over seeds
mean_agreements = mean_agreements.mean(axis=0)
mean_agreements

# %%
plt.style.use('seaborn-whitegrid')
# plot the agreement matrix
sns.set(style="whitegrid")
scorer_names = ["Least Confidence", "Entropy", "Margin", "Cross Entropy"]

fig, ax = plt.subplots(figsize=(10, 6))

cmap = sns.color_palette(["#4B2991", "#952EA0"], as_cmap=True)

sns.heatmap(mean_agreements, annot=True, xticklabels=scorer_names, yticklabels=scorer_names,
            # cmap=cmap, cbar=False, linewidths=.5,vmin=0.88, vmax=1.0)
            cmap=cmap, cbar=False, linewidths=.5, vmin=0.88, vmax=1.0)
# ax.set_title("Agreement Rate", fontsize=18)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.show()

# %%

flights = sns.load_dataset("flights")

flights_pv = flights.pivot("month", "year", "passengers")


sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(flights_pv, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(flights_pv, annot=True, fmt="d", linewidths=.5,
            cmap="YlGnBu", cbar_kws={"label": "Passengers"})
ax.set_title("Passenger Traffic by Month and Year", fontsize=18)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Month", fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), rotation=0)
plt.show()


# %%
