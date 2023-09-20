# %%
# import os
# os.chdir("../..")

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
from shell.fleet.fleet import Agent
from shell.fleet.data.data_utilization.data_utilize import *
from shell.fleet.data.recv import *

from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import logging
logging.basicConfig(level=logging.INFO)

# %%
seed_everything(0)

# %%
use_contrastive = True
num_tasks = 4

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
sender_dataset = get_dataset(**data_cfg)

# %%
net_cfg = {
    'depth': 4,
    'layer_size': 64,
    'num_init_tasks': num_tasks,
    'i_size': 28,
    'num_classes': 2,
    'num_tasks': 4,
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
sharing_cfg = DictConfig({
    "scorer": "cross_entropy",
    "num_queries": 4,
    "query_score_threshold": 0.0,
})
train_cfg = {
    # "num_epochs": 40,
    "num_epochs": 1,
}

agent = RecvDataAgent(0, 0, dataset,
                NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, 
                sharing_cfg)

sender = RecvDataAgent(1, 1, sender_dataset,
                NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, 
                sharing_cfg)



# %%
Xval, yval = agent.dataset.valset[0].tensors
print(Xval.shape)

# %%
# plot Xval on 2D using t-SNE and label with yval

Xval_embedded = TSNE(n_components=2, random_state=0).fit_transform(Xval.reshape(Xval.shape[0], -1))

plt.scatter(Xval_embedded[:, 0], Xval_embedded[:, 1], c=yval, cmap='tab10');
plt.legend()
plt.title("Raw image feature");

# %%
Xval.device

# %%
with torch.no_grad():
    agent.net.eval()
    Xval_untrain_embed = agent.net.encode(Xval.to(agent.net.device), task_id=0).detach().cpu()
    Xval_untrain_embedded = TSNE(n_components=2, random_state=0).fit_transform(Xval_untrain_embed)

agent.net.train()
plt.scatter(Xval_untrain_embedded[:, 0], Xval_untrain_embedded[:, 1], c=yval, cmap='tab10')
plt.legend()
plt.title("Untrained network feature");

# %%
for t in range(num_tasks):
    agent.train(t)
    sender.train(t)

# %%
with torch.no_grad():
    agent.net.eval()
    Xval_train_embed = agent.net.encode(Xval.to(agent.net.device), task_id=0).detach().cpu()
    Xval_train_embedded = TSNE(n_components=2, random_state=0).fit_transform(Xval_train_embed)

agent.net.train()
plt.scatter(Xval_train_embedded[:, 0], Xval_train_embedded[:, 1], c=yval, cmap='tab10')
plt.legend()
plt.title("Trained network feature");

# %%
qX, qY, ypred, scores = agent.compute_query(task_id=0, mode="current", debug_return=True)
qX = qX[0]
qY = qY[0]
ypred = ypred[0]
scores = scores[0]

q_plt = make_grid(qX)
plt.imshow(q_plt.permute(1, 2, 0))
# plotting the query
plt.title(f"{qY}, {ypred}, {scores}");

# %%
Xval_train_embed.shape

# %%
yval.shape

# %%
# plot the embedding with the query point ("hard" points)
with torch.no_grad():
    agent.net.eval()
    Xval_train_embed = agent.net.encode(Xval.to(agent.net.device), task_id=0).detach().cpu()
    qX_embed = agent.net.encode(qX.to(agent.net.device), task_id=0).detach().cpu()
    Xval_train_embedded = TSNE(n_components=2, random_state=0).fit_transform(Xval_train_embed)

agent.net.train()
plt.scatter(Xval_train_embedded[:, 0], Xval_train_embedded[:, 1], c=yval, cmap='tab10')
plt.scatter(qX_embed[:, 0], qX_embed[:, 1], c='r', marker='x', s=100)
plt.title("Trained network feature (receiver)");

# %%
plt.scatter(qX_embed[:, 0], qX_embed[:, 1], c='r', marker='x', s=100);

# %%
# plot the embedding with the query point ("hard" points)
with torch.no_grad():
    sender.net.eval()
    Xval_sender, yval_sender = sender.dataset.valset[0].tensors
    Xval_train_embed = sender.net.encode(Xval_sender.to(sender.net.device), task_id=0).detach().cpu()
    qX_embed = agent.net.encode(qX.to(sender.net.device), task_id=0).detach().cpu()
    Xval_train_embedded = TSNE(n_components=2, random_state=0).fit_transform(Xval_train_embed)

sender.net.train()
plt.scatter(Xval_train_embedded[:, 0], Xval_train_embedded[:, 1], c=yval_sender, cmap='tab10')
plt.scatter(qX_embed[:, 0], qX_embed[:, 1], c='r', marker='x', s=100)
plt.title("Trained network feature (sender)");

# %%
# neighbors = agent.nearest_neighbors(qX, neighbors=5, tasks=[0]) # (num_X, num_neighbors, 1, 28, 28)
neighbors = sender.nearest_neighbors(qX, neighbors=5,
                                     chosen_tasks=[0]) # (num_X, num_neighbors, 1, 28, 28)
neigbors = neighbors.cpu()
# plot the neighbors per row
n_X = neighbors.shape[0]
fig, ax = plt.subplots(n_X, 1, figsize=(20, 4))
for i in range(n_X):
    n_plt = make_grid(neighbors[i].cpu())
    ax[i].imshow(n_plt.permute(1, 2, 0))
    ax[i].axis('off')

# %%
qidx = []
for i, q in enumerate(qX):
    for idx, x in enumerate(Xval):
        if torch.equal(q, x):
            qidx.append(idx)
            break

# %%
Xval[qidx].shape

# %%
Xval_embedded = TSNE(n_components=2, random_state=0).fit_transform(Xval.reshape(Xval.shape[0], -1))

plt.scatter(Xval_embedded[:, 0], Xval_embedded[:, 1], c=yval, cmap='tab10');
plt.scatter(Xval_embedded[qidx, 0], Xval_embedded[qidx, 1], c='r', marker='x', s=100)
plt.legend()
plt.title("Raw image feature");


