'''
File: /data_aug.py
Project: learning-hive
Created Date: Wednesday March 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.datasets.datasets import get_dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from shell.utils.utils import seed_everything
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
data_cfg = {
    # "dataset_name": "mnist",
    # "dataset_name": "fashionmnist",
    "dataset_name": "cifar100",
    "num_tasks": 1,
    # "num_classes_per_task": 2,
    "num_train_per_task": -1,
    # "labels": np.array([1, 2]),
    'remap_labels': True,
}
dataset = get_dataset(**data_cfg)
