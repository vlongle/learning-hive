'''
File: /test_sharing_cont.py
Project: learning-hive
Created Date: Thursday March 23rd 2023
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
logging.basicConfig(level=logging.INFO)


seed_everything(0)
## ==================== ##
# Define dataset
data_cfg = {
    "dataset_name": "kmnist",
    "num_tasks": 1,
    # "num_train_per_task": 128,
    "num_train_per_task": -1,
    "labels": np.array([1, 2]),
    'remap_labels': True,
}
dataset = get_dataset(**data_cfg)


data_cfg2 = {
    "dataset_name": "kmnist",
    "num_tasks": 1,
    "num_train_per_task": 128,
    # "labels": np.array([3, 1]),
    "labels": np.array([1, 3]),
    'remap_labels': True,
}
dataset2 = get_dataset(**data_cfg2)

# FLIP THE LABELS!
# Y = 1.0 - dataset2.trainset[0].tensors[1]  # flipped the labels
Y = dataset2.trainset[0].tensors[1]
dataset2.trainset[0] = torch.utils.data.TensorDataset(
    dataset2.trainset[0].tensors[0], Y.long(),
    torch.full((len(Y),), 0, dtype=torch.long))

task_id = 0
testloaders = {task: torch.utils.data.DataLoader(testset,
                                                 batch_size=128,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}
trainloader1 = torch.utils.data.DataLoader(dataset.trainset[0],
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True,
                                           )
trainloader2 = torch.utils.data.DataLoader(dataset2.trainset[0],
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True,
                                           )
## ==================== ##


"""
Define NetCls and AgentCls
"""

net_cfg = {
    'depth': 2,
    'layer_size': 64,
    'num_init_tasks': -1,
    'i_size': 28,
    'num_classes': 2,
    'num_tasks': 1,
    'dropout': 0.0,
}

agent_cfg = {
    'memory_size': 64,
    'use_contrastive': True,
}
net = MLP(**net_cfg)
agent = NoComponentsER(net, **agent_cfg)

# somehow eval_net makes the model actually train like wtf???
# print(eval_net(net, testloaders))

# viz_embedding(net, testloaders, "init.png")

# # normal, local training
agent.train(trainloader1, task_id=0, num_epochs=100, testloaders=testloaders,
            # train_mode='both', save_freq=1)
            train_mode='both', save_freq=20)

# viz_embedding(net, testloaders, "after_training.png")

print('\n\n SHARING DATA \n\n')
# # train on newly shared data!

# TODO: interleave the training of my new dataset with my own to avoid catastrophic forgetting.

for i in range(20):
    agent._train(trainloader2, task_id=0, num_epochs=1, testloaders=testloaders,
                 save_freq=1,
                 train_mode='cl')  # NOTE: have to use primitive _train to avoid
    # mixing the shared data (with potentially different label semantic) with the
    # experience replay of agent.
    agent.train(trainloader1, task_id=0, num_epochs=1, testloaders=testloaders,
                save_freq=1,
                train_mode='cl')

# mega_testloaders = testloaders | {
#     1: torch.utils.data.DataLoader(dataset2.testset[0],
#                                    batch_size=128,
#                                    shuffle=False,
#                                    num_workers=0,
#                                    pin_memory=True,
#                                    )
# }
# viz_embedding(net, mega_testloaders, "after_sharing.png")

# agent._train(trainloader2, task_id=0, num_epochs=100, testloaders=testloaders,
#              save_freq=20,
#              train_mode='ce')

# print('\n\n Retraining\n\n')
# # retrain on my own data
# agent.train(trainloader1, task_id=0, num_epochs=20, testloaders=testloaders,
#             train_mode='both')

# viz_embedding(net, mega_testloaders, "after_retraining.png")
