# %%
import os

# %%
import omegaconf
from shell.utils.experiment_utils import *
from shell.utils.metric import *
import matplotlib.pyplot as plt
from shell.fleet.network import TopologyGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from shell.fleet.fleet import Agent, Fleet
from shell.fleet.data.data_utilize import *
from shell.fleet.data.recv import *

from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import logging
logging.basicConfig(level=logging.INFO)

# %%
use_contrastive = True
num_tasks = 4
num_init_tasks = 4
num_epochs = 2

# %%
seed_everything(0)

data_cfg = {
    "dataset_name": "mnist",
    "num_tasks": num_tasks,
    "num_train_per_task": 128,
    "num_val_per_task": 102,
    'remap_labels': True,
    'use_contrastive': use_contrastive,
}
dataset = get_dataset(**data_cfg)

# %%
seed_everything(7)
sender_dataset1 = get_dataset(**data_cfg)

# %%
seed_everything(9)
sender_dataset2 = get_dataset(**data_cfg)

# %%
net_cfg = {
    'depth': num_init_tasks,
    'layer_size': 64,
    'num_init_tasks': num_init_tasks,
    'i_size': 28,
    'num_classes': 2,
    'num_tasks': num_tasks,
    'dropout': 0.0,
}

agent_cfg = {
    'memory_size': 64,
    'use_contrastive': use_contrastive,
    'save_dir': 'test',
}

# %%
NetCls = MLPSoftLLDynamic
LearnerCls = CompositionalDynamicER
AgentCls = RecvDataAgent
sharing_cfg = DictConfig({
    "scorer": "cross_entropy",
    "num_queries": 5,
    'num_data_neighbors': 5,
    'num_filter_neighbors': 5,
    'num_coms_per_round': 2,
    "query_score_threshold": 0.0,
})
train_cfg = {
    # "num_epochs": 40,
    "num_epochs": num_epochs,
}

# %%
# create a graph of 3 nodes and 2 edges from 2 and 3 to 1
g = TopologyGenerator(num_nodes=3).generate_fully_connected()
TopologyGenerator.plot_graph(g)

# %%
fleet = Fleet(g, 0, [dataset, sender_dataset1, sender_dataset2], 
              sharing_cfg, AgentCls, NetCls, LearnerCls, net_cfg, agent_cfg, 
              train_cfg)

# %%
for t in range(num_tasks):
    fleet.train(t)

# %%
fleet.communicate(num_tasks-1)

# %%
fleet.agents[0].query

# %%



