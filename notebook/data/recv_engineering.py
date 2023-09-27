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
from shell.fleet.data.send import *

from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import logging
logging.basicConfig(level=logging.INFO)

# %%
seed_everything(0)

# %%
use_contrastive = True
num_tasks = 4
num_init_tasks = 2
dataset_name = "mnist"
bandwidth = 20

data_cfg = {
    "dataset_name": dataset_name,
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
    'num_tasks': 4,
    'dropout': 0.0,
}

agent_cfg = {
    'memory_size': 64,
    'use_contrastive': use_contrastive,
    'save_dir': 'test',
    'dataset_name': dataset_name,
}

# %%
NetCls = MLPSoftLLDynamic
LearnerCls = CompositionalDynamicER
AgentCls = SendDataAgent
sharing_cfg = DictConfig({
    "scorer": "cross_entropy",
    "num_queries": 5,
    'num_data_neighbors': 5,
    'num_filter_neighbors': 5,
    'num_coms_per_round': 2,
    "query_score_threshold": 0.0,
    "shared_memory_size": 50,
    "exploration_strategy": "epsilon_greedy",
    "min_epsilon": 0.1,
    "eps_decay_rate": 0.9,
    "init_epsilon": 0.8,
    "bandwidth": bandwidth,

})
train_cfg = {
    # "num_epochs": 40,
    "num_epochs": 1,
}


# %%
datasets = [dataset, sender_dataset1, sender_dataset2]
g = TopologyGenerator(num_nodes=len(datasets)).generate_fully_connected()
fleet = Fleet(g, 0, datasets,
              sharing_cfg, AgentCls, NetCls, LearnerCls, net_cfg, agent_cfg, 
              train_cfg)

# %%
receiver = fleet.agents[0]
sender = fleet.agents[1]

# %%
receiver.init_recommendation_engine()

# %%
for t in range(num_init_tasks):
    fleet.train(t)

# %%
receiver.prepare_data(task_id=t, neighbor_id=1)

# %%
gt_scores = torch.rand(size=(bandwidth,))

receiver.process_feedback(gt_scores, neighbor_id=1)

# %%



