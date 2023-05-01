'''
File: /send_utilize_ablation.py
Project: learning-hive
Created Date: Thursday April 27th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
# %%
from shell.fleet.data.send_utilize import *
from shell.fleet.fleet import Agent
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from shell.fleet.network import TopologyGenerator
import matplotlib.pyplot as plt
from shell.utils.metric import *
from shell.utils.experiment_utils import *
import omegaconf
import os
# %%

# %%
# save_root_dir = "vanilla_results"
save_root_dir = "vanilla_remove_datasets_hack_results"
dataset = "mnist"
algo = "modular"
num_train = 64
seed = 0
use_contrastive = True

# %%
job_name = f"{dataset}_{algo}_numtrain_{num_train}"
if use_contrastive:
    job_name += "_contrastive"
experiment = os.path.join(save_root_dir, job_name,
                          dataset, algo, f"seed_{seed}")

# %%
config_path = os.path.join(experiment, "hydra_out", ".hydra", "config.yaml")
# read the config file
cfg = omegaconf.OmegaConf.load(config_path)
cfg

# %%
graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg = setup_experiment(
    cfg)
len(datasets)

# %%
classes_sequence_list = [dataset.class_sequence for dataset in datasets]
classes_sequence_list

# %%
task_id = 3
num_added_components = None
receiver_id = 0
sender_id = 2

# %%
dataset = datasets[receiver_id]
testloaders = {task: torch.utils.data.DataLoader(testset,
                                                 batch_size=128,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}

# %%
receiver = Agent(receiver_id, seed, datasets[receiver_id],
                 NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg,
                 cfg.sharing_strategy)

sender = Agent(sender_id, seed, datasets[sender_id],
               NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg,
               cfg.sharing_strategy)

# %%
receiver.net = load_net(cfg, NetCls, net_cfg, agent_id=receiver_id,
                        task_id=task_id, num_added_components=num_added_components)
receiver.net

# %%
eval_net(receiver.net, testloaders)

# %%
num_classes_per_task = cfg.dataset.num_classes_per_task
recv_tasks = datasets[receiver_id].class_sequence[:(
    task_id + 1) * num_classes_per_task]
sender_task = datasets[sender_id].class_sequence[:(
    task_id + 1) * num_classes_per_task]
print(recv_tasks)
print(sender_task)

print(set(recv_tasks))
print(set(sender_task))

# %%
for t in range(task_id+1):
    sender_trainloader = torch.utils.data.DataLoader(datasets[sender_id].trainset[t],
                                                     batch_size=128,
                                                     shuffle=True,
                                                     num_workers=0,
                                                     pin_memory=True,
                                                     )
    sender.agent.update_multitask_cost(sender_trainloader, t)


for t in range(task_id+1):
    receiver_trainloader = torch.utils.data.DataLoader(datasets[receiver_id].trainset[t],
                                                       batch_size=128,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       )
    receiver.agent.update_multitask_cost(receiver_trainloader, t)

# %%
monodata = get_mono_dataset(sender.agent.memory_loaders[0].dataset, 0)
len(monodata)

monodata_true = get_ytrue_dataset(
    monodata, sender.dataset.class_sequence, sender.dataset.num_classes_per_task)


# %%
monodataremap = remapping(monodata_true, receiver.dataset.class_sequence,
                          receiver.dataset.num_classes_per_task, task_id=0)

print(monodataremap[0])
# %%
list(receiver.dataset.class_sequence[: 5])

global_utilize(monodata_true, receiver, task_id=2)
# %%
