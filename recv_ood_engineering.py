
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
from shell.fleet.data.data_utilize import *
import logging
logging.basicConfig(level=logging.INFO)

# %%
## TODO: for agent that has NO relevant data, we need to either
## 1) use contrastive only for shared buffer.
## 2) throw that away by distance thresholding.

# %%
# use_contrastive = False
use_contrastive = True
num_tasks = 4
num_init_tasks = 4
num_epochs = 100
comm_freq = num_epochs + 1
load_from_checkpoint = True
# prefilter_strategy = "raw_distance"
prefilter_strategy = "oracle"
# prefilter_strategy = "None"
# scorer = "random"
scorer = "cross_entropy"
# load_from_checkpoint = False

normalize=False
use_projector=False

# normalize=True
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
    'normalize': normalize,
    'use_contrastive': use_contrastive,
    'use_projector': use_projector,
}

agent_cfg = {
    'memory_size': 64,
    'use_contrastive': use_contrastive,
    # 'save_dir': f'recv_ood_engineering/use_contrastive_{use_contrastive}_normalize_embedding_results',
    'save_dir': f'recv_ood_engineering/use_contrastive_{use_contrastive}',
}

# %%
NetCls = MLPSoftLLDynamic
LearnerCls = CompositionalDynamicER
AgentCls = RecvDataAgent
sharing_cfg = DictConfig({
    "scorer": scorer,
    "num_queries": 5,
    'num_data_neighbors': 5,
    # 'num_filter_neighbors': 2,
    'num_filter_neighbors': 5,
    'num_coms_per_round': 2,
    "query_score_threshold": 0.0,
    "shared_memory_size": 50,
    "comm_freq": comm_freq,
    "prefilter_strategy": prefilter_strategy,
})
train_cfg = {
    # "num_epochs": 40,
    "num_epochs": num_epochs,
}

# %%
# create a graph of 3 nodes and 2 edges from 2 and 3 to 1
g = TopologyGenerator(num_nodes=3).generate_fully_connected()
# %%
fleet = Fleet(g, 0, [dataset, sender_dataset1, sender_dataset2], 
              sharing_cfg, AgentCls, NetCls, LearnerCls, net_cfg, agent_cfg, 
              train_cfg)

# # %%
# skip the training and just load the model if dir exists
if load_from_checkpoint:
    print("LOADING MODEL FROM CHECKPOINT")
    for agent in fleet.agents:
        agent.load_model_from_ckpoint(task_id=num_tasks-1)
else:
    for t in range(num_tasks):
        fleet.train_and_comm(t)

print("EVALUATING MODELS")


@torch.inference_mode()
def compute_contrastive_embedding_norm(net, X, task_id):
    X_emb = net.contrastive_embedding(X.to(net.device), task_id)
    return torch.norm(X_emb, dim=1)

# for agent_id, agent in enumerate(fleet.agents):
#     testloaders = {task: torch.utils.data.DataLoader(testset,
#                                                          batch_size=128,
#                                                          shuffle=False,
#                                                          num_workers=0,
#                                                          pin_memory=True,
#                                                          ) for task, testset in enumerate(agent.dataset.testset[:num_init_tasks])}
#     valsets = agent.dataset.valset[:num_init_tasks]
#     # valsets = fleet.agents[(agent_id+1) % len(fleet.agents)].dataset.valset[:num_init_tasks]
#     valloaders = {task: torch.utils.data.DataLoader(valset,
#                                                             batch_size=128,
#                                                             shuffle=False,
#                                                             num_workers=0,
#                                                             pin_memory=True,
#                                                             ) for task, valset in enumerate(valsets)}
#     print(eval_net(agent.net, testloaders))
#     print(eval_net(agent.net, valloaders))
#     for task, valset in enumerate(valsets):
#         X, y = valset.tensors
#         for cls in range(2): # classification of two classes. Y is 0 or 1.
#             X_cls = X[y == cls]
#             # print(f"task {task}, cls {cls}, score {compute_contrastive_embedding_norm(agent.net, X_cls, (task+1) % num_init_tasks).mean()}")
#             print(f"task {task}, cls {cls}, embedding norm {compute_contrastive_embedding_norm(agent.net, X_cls, task).mean()}")
        
#         # embed_dist = agent.compute_embedding_dist(X, X, task) # dim: num_samples x num_samples
#         # reduce to 2x2 where each entry is the mean embedding distance between class i and j 
#         # TODO
#             # Initialize a 2x2 matrix to store the mean distances
#         mean_distances = np.zeros((2, 2))

#         # Iterate over class pairs and compute mean distances
#         for i in range(2):
#             for j in range(2):
#                 # Extract samples for both classes
#                 X_i = X[y == i]
#                 X_j = X[y == j]
#                 # Compute distances between samples of class i and j
#                 distances = agent.compute_embedding_dist(X_i, X_j, task)
#                 # Calculate the mean distance and store it
#                 mean_distances[i, j] = distances.mean()
#         print('mean_distances\n', mean_distances)

#     print()


    # for t in range(num_init_tasks):
    #     print(agent.net.structure[t])
    # print()

def viz_query(X, y, y_pred, scores, path="test.pdf"):
    fig, axs = plt.subplots(len(scores), 1, figsize=(10, 10))
    for i in range(len(scores)):
        axs[i].imshow(X[i].permute(1, 2, 0))
        color = "green" if y[i] == y_pred[i] else "red"
        axs[i].set_title(f"{y[i]} / {y_pred[i]} / {scores[i]:.2f}", color=color) 
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)



for node in fleet.agents:
    for task in range(num_tasks):
        trainloader = torch.utils.data.DataLoader(node.dataset.trainset[task],
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    )
        node.agent.update_multitask_cost(trainloader, task)
        node.agent.T += 1 

receiver = fleet.agents[0]
# print(receiver.agent.T)
# print(receiver.agent.replay_buffers)
# exit(0)
fleet.communicate(task_id=num_tasks-1, start_com_round=0)


"""
For some reasons, normalize on MNIST is just much worse. We should look at the norm of OOD and IID samples, and plot their dist.

Norm: somehow implicitly being normalized to 0-1. Even OOD samples will be mapped to this range.

Also, compute the sim matrix on the valset and plot the barplot of same class vs diff class.
"""

# for agent in fleet.agents:
#     X, y, y_pred, scores = agent.compute_query(num_tasks-1, debug_return=True)
#     for task in X:
#         path = f"{agent.save_dir}/task_{task}/query_{scorer}.pdf"
#         viz(X[task], y[task], y_pred[task], scores[task], path=path)




# %%
# fleet.communicate(num_tasks-1)

# # %%
receiver = fleet.agents[0]

def compute_image_search_quality(node, neighbor, neighbor_id, task_id):
    """
    Return a num_classes x num_classes matrix where each entry[i, j]
    is the number of query images of class i that are given neighbors of class j.
    """
    Y_query = node.query_y[task_id]
    # Y_query_global.shape=(num_queries) where each entry is the global label of the query
    # range from 0 to num_classes 
    Y_query_global = get_global_labels(Y_query, [task_id] * len(Y_query), node.dataset.class_sequence, node.dataset.num_classes_per_task)

    Y_neighbor = node.incoming_extra_info[neighbor_id]['Y_neighbors'][task_id]
    task_neighbor = node.incoming_extra_info[neighbor_id]['task_neighbors'][task_id]
    print(Y_query_global)
    print(node.incoming_extra_info[neighbor_id]['task_neighbors_prefilter'][task_id])
    task_neighbors_prefilter = node.incoming_extra_info[neighbor_id]['task_neighbors_prefilter'][task_id]
    Y_neighbor_flatten = Y_neighbor.view(-1)
    task_neighbor_flatten = task_neighbor.view(-1)
    # print(Y_neighbor_flatten.shape, task_neighbor_flatten.shape)

    # Y_neighbor_global.shape=(num_queries, num_neighbors)
    # where each entry is the global label of the neighbor, range from 0 to num_classes
    Y_neighbor_global = get_global_labels(Y_neighbor_flatten, task_neighbor_flatten, neighbor.dataset.class_sequence, 
                     neighbor.dataset.num_classes_per_task).reshape(Y_neighbor.shape)


    # print(Y_query.shape, Y_neighbor.shape, task_neighbor.shape, Y_neighbor_global.shape)
    num_classes = len(np.unique(node.dataset.class_sequence))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(Y_query_global)):
        for j in range(Y_neighbor_global.shape[1]):  # Assuming Y_neighbor_global is a 2D array
            if task_neighbors_prefilter[i, j] == -1:
                continue
            query_label = Y_query_global[i]
            neighbor_label = Y_neighbor_global[i, j]
            confusion_matrix[query_label, neighbor_label] += 1
    
    print(confusion_matrix)
    acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    if np.isnan(acc):
        acc = 1.0

    # print('y_query_global', Y_query_global)

    X_neighbor = node.incoming_data[neighbor_id][task_id] # shape=(num_queries, num_neighbors, 1, 28, 28)
    viz_neighbor_data(X_neighbor, path=f"{node.save_dir}/task_{task_id}/neighbor_{neighbor_id}_{prefilter_strategy}.pdf")
    return confusion_matrix, acc

def viz_neighbor_data(X_neighbor, path):
    """
    Visualize and save the neighbor data.

    Parameters:
        X_neighbor (torch.Tensor): Neighbor data to visualize.
        path (str): Path to save the visualization as a PDF file.
    """
    num_queries, num_neighbors, _, height, width = X_neighbor.shape

    fig, axs = plt.subplots(num_queries, num_neighbors, figsize=(10, 10))
    for i in range(num_queries):
        for j in range(num_neighbors):
            axs[i, j].imshow(X_neighbor[i, j].permute(1, 2, 0))
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    # close the figure to save memory
    plt.close(fig)


# neighbor = fleet.agents[1]
accs = []
for task in range(num_tasks):
    for agent in fleet.agents:
        for other_agent in fleet.agents:
            if agent.node_id == other_agent.node_id:
                continue
            conf_mat, acc = compute_image_search_quality(agent, other_agent, other_agent.node_id, task)
            print(f"node {agent.node_id} task {task} neighbor {other_agent.node_id} acc {acc}")
            accs.append(acc)

print(f"mean acc {np.mean(accs)}")

# print('receiver incoming_query_extra_info', receiver.incoming_query_extra_info)
# print('receiver class_sequence', receiver.dataset.class_sequence)

neighbor_id = 1
qX = torch.cat(list(receiver.incoming_query[neighbor_id].values()), dim=0)
print(receiver.prefilter_oracle(qX, neighbor_id, n_filter_neighbors=3))
print(receiver.dataset.class_sequence)
print(fleet.agents[1].dataset.class_sequence)
print(fleet.agents[2].dataset.class_sequence)
# print('T', receiver.agent.T)
# print()

# inspect_task = 0

# # %%
# receiver.query[inspect_task].shape, receiver.incoming_data[1][inspect_task].shape, receiver.incoming_data[2][inspect_task].shape

# # %%
# query = receiver.query[inspect_task] # 5x1x28x28
# neighbor_data1 = receiver.incoming_data[1][inspect_task]
# neighbor_data2 = receiver.incoming_data[2][inspect_task]

# # %%
# plt.imshow(make_grid(query).permute(1,2,0).cpu().numpy());

# # %%
# plt.imshow(make_grid(neighbor_data1.view(-1,1,28,28)).permute(1, 2, 0));

# # %%
# # plot neighbor_data1 along with the query
# concat = torch.cat((query.unsqueeze(1), neighbor_data1), dim=1) # 5x6x1x28x28
# n_queries = query.shape[0]
# n_neighbors = neighbor_data2.shape[1]
# grid_image = make_grid(concat.view(-1, 1, 28, 28), nrow=n_neighbors+1).permute(1, 2, 0)
# plt.imshow(grid_image)
# plt.show()

# # %%
# # plot neighbor_data1 along with the query
# concat2 = torch.cat((query.unsqueeze(1), neighbor_data2), dim=1) # 5x6x1x28x28
# n_queries = query.shape[0]
# n_neighbors = neighbor_data2.shape[1]
# grid_image = make_grid(concat2.view(-1, 1, 28, 28), nrow=n_neighbors+1).permute(1, 2, 0)
# plt.imshow(grid_image)
# plt.show()

# # %%
# X,y,_ = receiver.agent.shared_replay_buffers[inspect_task].tensors
# # NOTE: some incorrect img searches here. idk if this is going to hurt perf.
# plt.imshow(make_grid(X).permute(1, 2, 0));

# # %%



