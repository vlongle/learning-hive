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
parser = argparse.ArgumentParser(
    description='Run experiment with a specified seed.')

parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    # config = {
    #     "algo": ["monolithic", "modular"],
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
    #     "root_save_dir": "vanilla_results",
    #     "agent.use_contrastive": [True, False],
    #     "agent.memory_size": 32,
    #     "dataset": ["mnist", "kmnist", "fashionmnist"],
    # }

    # run_experiment(config)

    # # # === CNN experiments: CIFAR100 ===
    config = {
        # "algo": ["monolithic", "modular"],
        "algo": "modular",
        "seed": 0,
        "num_agents": 1,
        "parallel": True,
        "dataset": "cifar100",
        "dataset.num_trains_per_class": 256,
        "dataset.num_vals_per_class": -1,
        "dataset.remap_labels": True,
        "dataset.with_replacement": False,
        # "dataset.num_tasks": 20,
        "dataset.num_tasks": 6,
        "net": "cnn",
        "net.depth": 4,
        "num_init_tasks": 4,
        "net.dropout": 0.5,
        "train.num_epochs": args.num_epochs,
        "train.component_update_freq": args.num_epochs,
        "agent.memory_size": 64,
        "agent.batch_size": args.batch_size,
        "train.save_freq": 10,
        "agent.use_contrastive": False,
        "root_save_dir": f"experiment_results/monday_debug_cifar100_batch_{args.batch_size}_num_epochs_{args.num_epochs}",
    }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
