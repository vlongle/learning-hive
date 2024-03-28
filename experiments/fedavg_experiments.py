'''
File: /grad_experiments.py
Project: experiments
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import time
import datetime
from shell.utils.experiment_utils import run_experiment

import argparse
from shell.utils.utils import on_desktop

parser = argparse.ArgumentParser(
    description='Run experiment with a specified seed.')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
# parser.add_argument('--comm_freq', type=int, default=10)
parser.add_argument('--comm_freq', type=int, default=1)
# parser.add_argument('--when_reoptimize_structure', type=str,
#                     default="never", choices=["never", "always", "final"])
# parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "monolithic", "modular"], help='Algorithm for the experiment.')
args = parser.parse_args()

if on_desktop():
    prefix = ""
else:
    prefix = "/mnt/kostas-graid/datasets/vlongle/"


if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===

    # ACTUAL CONFIG
    num_init_tasks = 4
    num_tasks = 10
    # num_tasks = 5
    num_epochs = 100
    # num_epochs = 10
    # comm_freq = 1
    # comm_freq = 50 # how many epochs does a round of communication take place

    batch_size = 64
    save_freq = 1

    seed = args.seed
    dataset = args.dataset

    config = {
        # "algo": ["monolithic", "modular"],
        "algo": args.algo,
        "seed": args.seed,
        "seed": [0, 1, 2, 3, 4, 5, 6, 7],
        # "seed": [0, 1, 2, 3],
        "parallel": True,
        # "parallel": False,
        # "num_agents": 8,
        "num_agents": 20,
        "dataset": "combined",
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": num_tasks,
        "net": "mlp",
        "net.depth": 4,
        "num_init_tasks": num_init_tasks,
        "net.dropout": 0.5,
        "train.num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "train.init_num_epochs": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.save_freq": 10,
        "agent.use_contrastive": False,
        "agent.memory_size": 32,
        "root_save_dir": prefix + f"budget_experiment_results/jorge_setting_fedavg/comm_freq_{args.comm_freq}",
        # ================================================
        # GRAD SHARING SETUP
        "sharing_strategy": "grad_sharing",
        "sharing_strategy.num_coms_per_round": 1,
        "sharing_strategy.comm_freq": args.comm_freq,
        # "sharing_strategy.log_freq": 10,

        # ================================================
    }

    # config = {
    #     "algo": args.algo,
    #     "seed": args.seed,
    #     "num_agents": 8,
    #     "parallel": True,
    #     "dataset": "cifar100",
    #     "dataset.num_trains_per_class": 256,
    #     "dataset.num_vals_per_class": -1,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": False,
    #     "net": "cnn",
    #     "net.depth": 4,
    #     "num_init_tasks": 4,
    #     "dataset.num_tasks": 20,
    #     "net.dropout": 0.5,
    #     "train.init_num_epochs": num_epochs,
    #     "train.init_component_update_freq": num_epochs,
    #     "train.num_epochs": num_epochs,
    #     "train.component_update_freq": num_epochs,
    #     "agent.memory_size": 32,
    #     "agent.batch_size": 64,
    #     "train.save_freq": 10,
    #     "agent.use_contrastive": False,
    #     'net.no_sparse_basis': True,

    #     "sharing_strategy": "grad_sharing",
    #     "sharing_strategy.num_coms_per_round": 1,
    #     "sharing_strategy.comm_freq": 5,
    #     "root_save_dir": prefix + f"experiment_results/jorge_setting_fedavg",
    # }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
