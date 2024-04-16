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
                    "mnist", "kmnist", "fashionmnist", "cifar100", "combined"], help='Dataset for the experiment.')
parser.add_argument('--comm_freq', type=int, default=5)
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "monolithic", "modular"], help='Algorithm for the experiment.')
parser.add_argument('--topology', type=str, default='fully_connected')
parser.add_argument('--edge_drop_prob', type=float, default=0.0)

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
    num_epochs = 100

    batch_size = 64
    save_freq = 1

    seed = args.seed

    num_agents = 20 if args.dataset == "combined" else 8
    if args.dataset != "cifar100":
        config = {
            "algo": args.algo,
            "seed": [0, 1, 2, 3, 4, 5, 6, 7],
            # "seed": args.seed,
            "dataset": args.dataset,
            "num_agents": num_agents,
            "parallel": True,
            # "topology": args.topology,
            "topology": ['ring', 'server'],
            "edge_drop_prob": args.edge_drop_prob,

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
            'net.no_sparse_basis': True,
            "train.save_freq": 10,
            "agent.use_contrastive": False,
            "agent.memory_size": 32,
            "root_save_dir": prefix + f"rerun_topology_experiment_results/jorge_setting_fedavg/comm_freq_{args.comm_freq}/topology_{args.topology}_edge_drop_{args.edge_drop_prob}",
            # ================================================
            # GRAD SHARING SETUP
            "sharing_strategy": "grad_sharing",
            "sharing_strategy.num_coms_per_round": 1,
            "sharing_strategy.comm_freq": args.comm_freq,
            # "sharing_strategy.log_freq": 10,

            # ================================================
        }

    else:
        config = {
            "algo": args.algo,
            "seed": [0, 1, 2, 3, 4, 5, 6, 7],
            "num_agents": 8,
            "parallel": True,

            "topology": args.topology,
            "edge_drop_prob": args.edge_drop_prob,

            "dataset": "cifar100",
            "dataset.num_trains_per_class": 256,
            "dataset.num_vals_per_class": -1,
            "dataset.remap_labels": True,
            "dataset.with_replacement": False,
            "net": "cnn",
            "net.depth": 4,
            "num_init_tasks": 4,
            "dataset.num_tasks": 20,
            "net.dropout": 0.5,
            "train.init_num_epochs": num_epochs,
            "train.init_component_update_freq": num_epochs,
            "train.num_epochs": num_epochs,
            "train.component_update_freq": num_epochs,
            "agent.memory_size": 32,
            "agent.batch_size": 64,
            "train.save_freq": 10,
            "agent.use_contrastive": False,
            'net.no_sparse_basis': True,

            "sharing_strategy": "grad_sharing",
            "sharing_strategy.num_coms_per_round": 1,
            "sharing_strategy.comm_freq": args.comm_freq,
            # "root_save_dir": prefix + f"budget_experiment_results/jorge_setting_fedavg/comm_freq_{args.comm_freq}",
            "root_save_dir": prefix + f"topology_experiment_results/jorge_setting_fedavg/comm_freq_{args.comm_freq}/topology_{args.topology}_edge_drop_{args.edge_drop_prob}",
        }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
