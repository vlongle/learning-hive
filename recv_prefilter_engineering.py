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
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO)

use_contrastive = True
num_tasks = 4
num_init_tasks = 4
num_epochs = 100
comm_freq = num_epochs + 1
load_from_checkpoint = True
prefilter_strategy = "None"
scorer = "cross_entropy"
normalize=False
use_projector=False

def load_models(fleet):
    if load_from_checkpoint:
        print("LOADING MODEL FROM CHECKPOINT")
        for agent in fleet.agents:
            agent.load_model_from_ckpoint(task_id=num_tasks-1)
    else:
        for t in range(num_tasks):
            fleet.train_and_comm(t)

def update_relay_buffers(fleet):
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


@torch.inference_mode()
def compute_contrastive_embedding_norm(net, X, task_id):
    X_emb = net.contrastive_embedding(X.to(net.device), task_id)
    return torch.norm(X_emb, dim=1)

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

seed_everything(7)
sender_dataset1 = get_dataset(**data_cfg)

seed_everything(9)
sender_dataset2 = get_dataset(**data_cfg)

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
    'save_dir': f'recv_ood_engineering/use_contrastive_{use_contrastive}',
}

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
    "num_epochs": num_epochs,
}

g = TopologyGenerator(num_nodes=3).generate_fully_connected()
fleet = Fleet(g, 0, [dataset, sender_dataset1, sender_dataset2], 
              sharing_cfg, AgentCls, NetCls, LearnerCls, net_cfg, agent_cfg, 
              train_cfg)


load_models(fleet)
update_relay_buffers(fleet)

fleet.communicate(task_id=num_tasks-1, start_com_round=0)
receiver = fleet.agents[0]
# print(receiver.incoming_query[1][0].shape)
# print(receiver.incoming_query_extra_info[1]['query_global_y'][0].shape)


def shape_query_format(fleet):
    for agent in fleet.agents:
        agent_X, agent_y = [], []
        for neighbor_id in agent.incoming_query:
            query_global_y = agent.incoming_query_extra_info[neighbor_id]['query_global_y'] # dict of task_id -> global_y
            query_global_y = torch.cat(list(query_global_y.values()), dim=0) # shape=(num_queries)
            concat_query = torch.cat(list(agent.incoming_query[neighbor_id].values()), dim=0) # concat_query = size (N, C, H, W)
            agent_X.append(concat_query)
            agent_y.append(query_global_y)
        agent_X = torch.cat(agent_X, dim=0)
        agent_y = torch.cat(agent_y, dim=0)
        agent.agent_X = agent_X
        agent.agent_y = agent_y

        # agent.agent_local_labels = {}
        # for task in range(num_tasks):
        #     agent.agent_local_labels[task] = get_local_labels_for_task(agent.agent_y, task, agent.dataset.class_sequence, agent.dataset.num_classes_per_task)


computer_type = "margin"

@torch.inference_mode()
def prefilter_computer(net, X, task_id, computer_type, reduce=True):
    """
    NOTE: these are based on the classifier. TODO: try based on the contrastive embedding.
    """
    logits = net(X.to(net.device), task_id)
    if computer_type == "cross_entropy":
        scores = entropy_scorer(logits)
    elif computer_type == "least_confidence":
        scores = least_confidence_scorer(logits)
    elif computer_type == "margin":
        scores = margin_scorer(logits)
    elif computer_type == "random":
        scores = random_scorer(logits)
    else:
        raise NotImplementedError

    scores = -1 * scores.cpu()
    if reduce:
        scores = torch.mean(scores).item()
    return scores


def oracle_label(global_Y, task, class_sequence, num_classes_per_task):
    return get_local_labels_for_task(global_Y, task, class_sequence, num_classes_per_task)

def compute_prefilter_score(fleet, computer):
    for agent in fleet.agents:
        scores = []
        for task in range(num_tasks):
            y = oracle_label(agent.agent_y, task, agent.dataset.class_sequence, agent.dataset.num_classes_per_task)
            X = agent.agent_X
            # ood is where y == -1, in-distribution is where y != -1
            ood_X = X[y == -1]
            in_dist_X = X[y != -1]
            scores.append([computer(agent.net, ood_X, task, computer_type), computer(agent.net, in_dist_X, task,computer_type)])
        agent.prefilter_scores = torch.tensor(scores)
        agent.mean_prefilter_scores = torch.mean(agent.prefilter_scores, dim=0)
    
    ood_score, in_dist_score = [], []
    for agent in fleet.agents:
        ood_score.append(agent.mean_prefilter_scores[0].item())
        in_dist_score.append(agent.mean_prefilter_scores[1].item())
    return ood_score, in_dist_score


shape_query_format(fleet)
ood_score, in_dist_score = compute_prefilter_score(fleet, prefilter_computer)
print(ood_score, in_dist_score)
print(np.mean(ood_score), np.mean(in_dist_score))

for agent in fleet.agents:
    print('agent_id', agent.node_id, 'prefilter_scores', agent.prefilter_scores)

def classify_ood_id(agent, task_id, threshold, computer_type):
    """
    Classifies each data point in agent.agent_X as OOD or ID based on the score from prefilter_computer.
    """
    scores = prefilter_computer(agent.net, agent.agent_X, task_id, computer_type,reduce=False)
    print('agent_id', agent.node_id, 'task_id', task_id, 'scores', scores)
    # Classify as OOD if score is below the threshold, otherwise ID
    classifications = (scores < threshold).int()
    return classifications

def calculate_metrics(classifications, ground_truth):
    """
    Calculates the accuracy and F-1 score of the OOD/ID classifications against the oracle ground truth.
    """
    correct = (classifications == ground_truth).sum().item()
    total = ground_truth.numel()
    accuracy = correct / total

    # Compute F-1 score using sklearn's utility function
    f1 = f1_score(ground_truth.cpu(), classifications.cpu())

    return accuracy, f1

def get_ood_id_ratio(ground_truth):
    """
    Calculates the ratio of OOD to ID samples.
    """
    num_ood = (ground_truth == 1).sum().item()
    num_id = (ground_truth == 0).sum().item()
    ratio = num_ood / (num_ood + num_id) 
    return ratio


# Example Usage
threshold = 0.6  # Set an appropriate threshold based on your observations
# task_id = 0      # Example task ID
for agent in fleet.agents:
    for task_id in range(num_tasks):
        # Oracle labels (ground truth)
        oracle_labels = oracle_label(agent.agent_y, task_id, agent.dataset.class_sequence, agent.dataset.num_classes_per_task)
        # Convert oracle labels to binary (0 for ID, 1 for OOD)
        ground_truth = (oracle_labels == -1).int()

        # Classify and calculate accuracy
        classifications = classify_ood_id(agent, task_id, threshold, computer_type)
        print('gt', ground_truth)
        accuracy, f1 = calculate_metrics(classifications, ground_truth)
        ood_id_ratio = get_ood_id_ratio(ground_truth)
        print(f"Agent {agent.node_id} on Task {task_id}:")
        print(f"OOD/ID Ratio: {ood_id_ratio:.2f}, Accuracy: {accuracy:.2f}, F-1 Score: {f1:.2f}")



# print(fleet.agents[0].agent_X.shape, fleet.agents[0].agent_y.shape)
# print(fleet.agents[1].agent_X.shape, fleet.agents[1].agent_y.shape)
# print(fleet.agents[2].agent_X.shape, fleet.agents[2].agent_y.shape)
# print(receiver.agent_local_labels)
# print(receiver.agent_local_labels[0].shape)