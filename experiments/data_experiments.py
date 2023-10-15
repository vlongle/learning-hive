'''
File: /experiments.py
Project: experiments
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
'''
File: /experiments.py
Project: experiments
Created Date: Thursday March 16th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import time
import datetime
from shell.utils.experiment_utils import run_experiment

import argparse
parser = argparse.ArgumentParser(description='Run experiment with a specified seed.')
parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
args = parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    # config = {
    #     "algo": ["monolithic", "modular"],
    #     # "seed": [0, 1, 2, 3],
    #     "seed": 0,
    #     "parallel": True,
    #     "num_agents": 8,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": 10,
    #     "net": "mlp",
    #     "net.depth": 4,
    #     "num_init_tasks": 4,
    #     "net.dropout": 0.0,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "train.save_freq": 20,
    #     "root_save_dir": "recv_results",
    #     "agent.use_contrastive": True,
    #     "agent.memory_size": 32,
    #     "sharing_strategy": "recv_data",
    #     "dataset": ["mnist", "kmnist", "fashionmnist"],
    # }

    seed = args.seed

    num_epochs = 100
    num_init_tasks = 4
    num_tasks = 10
    batch_size = 64

    dataset = args.dataset


    # small debug experiment
    config = {
        "algo": ["monolithic"],
        "seed": 0,
        "parallel": True,
        # "parallel": False,
        "num_agents": 8,
        "dataset": dataset,
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": num_tasks,
        "net": "mlp",
        "net.depth": num_init_tasks,
        "num_init_tasks": num_init_tasks,
        "net.dropout": 0.0,
        "train.num_epochs": num_epochs,
        "train.init_num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.save_freq": 20,
        "root_save_dir": "experiment_results/recv_results",
        "agent.use_contrastive": True,
        "agent.memory_size": 32,
        "sharing_strategy": "recv_data",
        "agent.batch_size": batch_size,
    }
    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
