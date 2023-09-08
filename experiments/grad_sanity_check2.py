'''
File: /grad_sanity_check2.py
Project: experiments
Created Date: Thursday September 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
'''
File: /grad_sanity_check.py
Project: experiments
Created Date: Tuesday September 5th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

"""
1. Does the normal averaging destroy the structure of learned weights?
2. Under the toy domain, the fisher fixes that?

Settings: 2 agents with 2 tasks. We'd like to basically distill into one model that solves both tasks.

Checklist
[] Copying works.
[] See how long takes to bounce back
[] fischer info

"""


from shell.fleet.grad.fisher_monograd import ModelFisherSyncAgent
from fisher import EWC
from copy import deepcopy
from shell.datasets.datasets import get_dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from shell.utils.utils import seed_everything, viz_embedding
import torch
import subprocess
import torch.nn as nn
import torch
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
import pickle
logging.basicConfig(level=logging.INFO)

"""
Two agents trained separately
"""

num_tasks = 10

net_cfg = {
    'depth': 2,
    'layer_size': 64,
    'num_init_tasks': 2,
    'i_size': 28,
    'num_classes': 2,
    'num_tasks': num_tasks,
    'dropout': 0.0,
}

agent_cfg = {
    'memory_size': 64,
    'use_contrastive': False,
    'save_dir': '',
}


data_cfg = {
    "dataset_name": "mnist",
    "num_tasks": num_tasks,
    "num_train_per_task": 128,
    "num_val_per_task": 102,
    'remap_labels': True,
    'use_contrastive': False,
    "with_replacement": True,
}
print(num_tasks)
bob_dataset = get_dataset(**data_cfg)
print(len(bob_dataset.trainset))

# # alice dataset is the same as bob's but tasks 0,1 are switched with 2,3
alice_dataset = deepcopy(bob_dataset)
alice_dataset.trainset[0], alice_dataset.trainset[2] = alice_dataset.trainset[2], alice_dataset.trainset[0]
alice_dataset.trainset[1], alice_dataset.trainset[3] = alice_dataset.trainset[3], alice_dataset.trainset[1]
alice_dataset.testset[0], alice_dataset.testset[2] = alice_dataset.testset[2], alice_dataset.testset[0]
alice_dataset.testset[1], alice_dataset.testset[3] = alice_dataset.testset[3], alice_dataset.testset[1]
alice_dataset.valset[0], alice_dataset.valset[2] = alice_dataset.valset[2], alice_dataset.valset[0]
alice_dataset.valset[1], alice_dataset.valset[3] = alice_dataset.valset[3], alice_dataset.valset[1]
alice_dataset.class_sequence[0], alice_dataset.class_sequence[
    2] = alice_dataset.class_sequence[2], alice_dataset.class_sequence[0]
alice_dataset.class_sequence[1], alice_dataset.class_sequence[
    3] = alice_dataset.class_sequence[3], alice_dataset.class_sequence[1]

# alice = CompositionalDynamicER(MLPSoftLLDynamic(**net_cfg), **agent_cfg)
# bob = CompositionalDynamicER(MLPSoftLLDynamic(**net_cfg), **agent_cfg)

train_cfg = {'num_epochs': 10}
sharing_cfg = {}
alice = ModelFisherSyncAgent(0, 0, alice_dataset, MLPSoftLLDynamic, CompositionalDynamicER, net_cfg, agent_cfg,
                             train_cfg, sharing_cfg)

bob = ModelFisherSyncAgent(1, 0, alice_dataset, MLPSoftLLDynamic, CompositionalDynamicER, net_cfg, agent_cfg,
                           train_cfg, sharing_cfg)

# synchronizing the random projection to make sure they're the same.
alice.net.random_linear_projection.weight = bob.net.random_linear_projection.weight
alice.net.random_linear_projection.bias = bob.net.random_linear_projection.bias


alice.train(0)
alice.train(1)

alice.model = alice.prepare_model()
fisher = alice.prepare_fisher_diag()

print(fisher)

# # training ...
# for task_id in range(2):
#     testloaders = {task: torch.utils.data.DataLoader(testset,
#                                                      batch_size=128,
#                                                      shuffle=False,
#                                                      num_workers=0,
#                                                      pin_memory=True,
#                                                      ) for task, testset in enumerate(bob_dataset.testset[:(task_id+1)])}

#     trainloader = torch.utils.data.DataLoader(bob_dataset.trainset[task_id],
#                                               batch_size=64,
#                                               shuffle=True,
#                                               num_workers=0,
#                                               pin_memory=True,
#                                               )
#     valloader = torch.utils.data.DataLoader(bob_dataset.valset[task_id],
#                                             batch_size=256,
#                                             shuffle=False,
#                                             num_workers=4,
#                                             pin_memory=True,
#                                             )

#     bob.train(trainloader, task_id=task_id, num_epochs=10, testloaders=testloaders,
#               valloader=valloader, save_freq=1)


# print('\n\n\n')
# for task_id in range(2):
#     testloaders = {task: torch.utils.data.DataLoader(testset,
#                                                      batch_size=128,
#                                                      shuffle=False,
#                                                      num_workers=0,
#                                                      pin_memory=True,
#                                                      ) for task, testset in enumerate(alice_dataset.testset[:(task_id+1+2)])}

#     trainloader = torch.utils.data.DataLoader(alice_dataset.trainset[task_id],
#                                               batch_size=64,
#                                               shuffle=True,
#                                               num_workers=0,
#                                               pin_memory=True,
#                                               )
#     valloader = torch.utils.data.DataLoader(alice_dataset.valset[task_id],
#                                             batch_size=256,
#                                             shuffle=False,
#                                             num_workers=4,
#                                             pin_memory=True,
#                                             )

#     alice.train(trainloader, task_id=task_id, num_epochs=10, testloaders=testloaders,
#                 valloader=valloader, save_freq=1)

# # print(alice.net)


# # manually plug bob weights into alice...
# alice.net.add_tmp_module(task_id=2)
# alice.net.add_tmp_module(task_id=2)

# # turn the dict testloaders into one combined_testloader
# # combined_testloader = torch.utils.data.DataLoader(
# #     torch.utils.data.ConcatDataset(
# #         [testset.dataset for testset in testloaders.values()]),
# #     batch_size=128,
# #     shuffle=False,
# #     num_workers=0,
# #     pin_memory=True,
# # )

# alice_testloaders = {task: torch.utils.data.DataLoader(testset,
#                                                        batch_size=128,
#                                                        shuffle=False,
#                                                        num_workers=0,
#                                                        pin_memory=True,
#                                                        ) for task, testset in enumerate(alice_dataset.testset[:(task_id+1)])}
# print(alice_testloaders)
# ewc = EWC(alice.net, alice_testloaders)
# print("precision matrices")
# print(ewc._precision_matrices)

# """
# TODO: re-examine EWC.
# """

# # # print(alice.net)
# # alice.net.components[2].weight = bob.net.components[0].weight
# # alice.net.components[3].weight = bob.net.components[1].weight

# # alice.net.components[2].bias = bob.net.components[0].bias
# # alice.net.components[3].bias = bob.net.components[1].bias

# # # bob has 2 modules and alice has 4 modules. Black out modules 1 and 2
# # # of alice, and copy over the structure of bob's modules 1 and 2 into alice's modules 3 and 4
# # # 1. set all structure to -inf
# # alice.net.structure[2].data = -np.inf * torch.ones(4, 2)
# # alice.net.structure[3].data = -np.inf * torch.ones(4, 2)

# # # 2. copy over the structure of bob's modules 1 and 2 into alice's modules 3 and 4
# # alice.net.structure[2].data[2:, :] = bob.net.structure[0].data
# # alice.net.structure[3].data[2:, :] = bob.net.structure[1].data

# # # need to copy over the decoder as well
# # alice.net.decoder[2].weight = bob.net.decoder[0].weight
# # alice.net.decoder[3].weight = bob.net.decoder[1].weight
# # alice.net.decoder[2].bias = bob.net.decoder[0].bias
# # alice.net.decoder[3].bias = bob.net.decoder[1].bias

# # # print(alice.net.structure[2])
# # print(eval_net(alice.net, testloaders))
