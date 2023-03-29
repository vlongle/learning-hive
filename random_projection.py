'''
File: /random_projection.py
Project: learning-hive
Created Date: Friday March 24th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.datasets.datasets import get_dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from shell.utils.utils import seed_everything, viz_embedding
import torch
import subprocess
import torch.nn as nn
import torch
import os
from omegaconf import DictConfig
from shell.datasets.datasets import get_dataset
from shell.utils.utils import seed_everything
from pprint import pprint
from shell.fleet.network import TopologyGenerator
from shell.models.cnn_soft_lifelong_dynamic import CNNSoftLLDynamic
from shell.models.cnn import CNN
from shell.models.mlp import MLP
from shell.models.mlp_soft_lifelong_dynamic import MLPSoftLLDynamic
from shell.learners.er_dynamic import CompositionalDynamicER
from shell.learners.er_nocomponents import NoComponentsER
from shell.utils.experiment_utils import eval_net
from shell.utils.experiment_utils import setup_experiment
from sklearn.manifold import TSNE
import logging
import seaborn as sns
from shell.fleet.recv import least_confidence_scorer, entropy_scorer, margin_scorer, cross_entropy_scorer
logging.basicConfig(level=logging.INFO)

seed_everything(0)
data_cfg = {
    "dataset_name": "kmnist",
    "num_tasks": 4,
    "num_train_per_task": -1,
    "num_val_per_task": 102,
    "labels": np.array([1, 2, 3, 4, 5, 6, 5, 6]),
    'remap_labels': True,
}
dataset = get_dataset(**data_cfg)


net_cfg = {
    'i_size': 28,
    'layer_size': 64,
    'depth': 2,
    'num_init_tasks': 2,
    'num_classes': 2,
    'num_tasks': 1,
    'dropout': 0.0,
    "num_tasks": 4,
}

agent_cfg = {
    'memory_size': 64,
    'use_contrastive': True,
}

# net = MLP(**net_cfg)
# agent = NoComponentsER(net, **agent_cfg)


net = MLPSoftLLDynamic(**net_cfg)
agent = CompositionalDynamicER(net, **agent_cfg)

for task_id in range(4):
    testloaders = {task: torch.utils.data.DataLoader(testset,
                                                     batch_size=128,
                                                     shuffle=False,
                                                     num_workers=0,
                                                     pin_memory=True,
                                                     ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}

    trainloader = torch.utils.data.DataLoader(dataset.trainset[task_id],
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=0,
                                              pin_memory=True,
                                              )
    valloader = torch.utils.data.DataLoader(dataset.valset[task_id],
                                            batch_size=128,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            )
    if task_id == 3:
        print("TRAINING ON THE NEXT TASKS!\n\n")
    agent.train(trainloader, task_id=task_id, num_epochs=20, testloaders=testloaders,
                save_freq=1,
                train_mode='both',
                valloader=valloader,)
