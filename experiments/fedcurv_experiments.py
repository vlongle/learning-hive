'''
File: /fedcurv_experiments.py
Project: experiments
Created Date: Friday April 19th 2024
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2024 Long Le
'''


import time
import datetime
from shell.utils.experiment_utils import run_experiment
from shell.utils.utils import on_desktop

import argparse

parser = argparse.ArgumentParser(
    description='Run experiment with a specified seed.')

parser.add_argument('--algo', type=str, default="modular", choices=[
                    "monolithic", "modular"], help='Algorithm for the experiment.')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100", "combined"], help='Dataset for the experiment.')
parser.add_argument('--comm_freq', type=int, default=5)
parser.add_argument('--mu', type=float, default=0.0)
args = parser.parse_args()


if on_desktop():
    prefix = ""
else:
    prefix = "/mnt/kostas-graid/datasets/vlongle/learning_hive/"


if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===

    # ACTUAL CONFIG
    num_init_tasks = 4
    num_tasks = 10
    batch_size = 64
    num_epochs = 100
    memory_size = 32

    num_agents = 20 if args.dataset == "combined" else 8

    root_save_dir = prefix + \
        f"more_fl/fedcurv_mu_{args.mu}_comm_freq_{args.comm_freq}"

    config = {
        "algo": args.algo,
        "agent.batch_size": batch_size,
        # "seed": args.seed,
        "seed": [0, 1, 2, 3, 4, 5, 6, 7],
        "parallel": True,
        "num_agents": num_agents,
        "dataset": args.dataset,
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": num_tasks,
        "net": "mlp",
        "net.depth": num_init_tasks,
        "num_init_tasks": num_init_tasks,
        "net.dropout": 0.5,
        "train.num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "train.init_num_epochs": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.save_freq": 10,
        "agent.use_contrastive": False,
        "agent.memory_size": memory_size,
        "net.no_sparse_basis": True,

        "root_save_dir": root_save_dir,
        "sharing_strategy": "grad_sharing_curv",
        "sharing_strategy.num_coms_per_round": 1,
        "sharing_strategy.comm_freq": args.comm_freq,
        "sharing_strategy.mu": args.mu,
    }

    run_experiment(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
