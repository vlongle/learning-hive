'''
File: /debug_modular.py
Project: learning-hive
Created Date: Wednesday March 22nd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import torch
from shell.datasets.datasets import MNIST
from shell.utils.utils import seed_everything, create_dir_if_not_exist
from shell.learners.er_dynamic import CompositionalDynamicER
from shell.models.mlp_soft_lifelong_dynamic import MLPSoftLLDynamic
import logging

logging.basicConfig(level=logging.INFO)

seed_everything(1)


dataset = MNIST(num_tasks=10, with_replacement=True,
                num_train_per_task=256, num_val_per_task=50,
                remap_labels=False,)

# print(len(dataset.trainset[0]))
# print(dataset.trainset[0].tensors[1])

net = MLPSoftLLDynamic(i_size=28, layer_size=64, depth=4,
                       num_classes=10, num_tasks=10, num_init_tasks=4,
                       )
agent = CompositionalDynamicER(net, memory_size=64)


batch_size = 64

for task_id in range(10):
    trainloader = (
        torch.utils.data.DataLoader(dataset.trainset[task_id],
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    ))
    testloaders = {task: torch.utils.data.DataLoader(testset,
                                                     batch_size=128,
                                                     shuffle=False,
                                                     num_workers=0,
                                                     pin_memory=True,
                                                     ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}
    valloader = torch.utils.data.DataLoader(dataset.valset[task_id],
                                            batch_size=128,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            )
    agent.train(trainloader, task_id, testloaders=testloaders,
                valloader=valloader, component_update_freq=200, num_epochs=200, save_freq=1)
