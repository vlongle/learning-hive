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

parser = argparse.ArgumentParser(description='Run experiment with a specified seed.')
parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
parser.add_argument('--comm_freq', type=int, default=10)
parser.add_argument('--when_reoptimize_structure', type=str, default="never", choices=["never", "always", "final"])
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--use_contrastive', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===

    # ACTUAL CONFIG
    num_init_tasks = 4
    num_tasks = 5
    # num_epochs = 4
    # comm_freq = 1
    # comm_freq = 50 # how many epochs does a round of communication take place

    batch_size = 64
    save_freq = 1
    
    seed = args.seed
    dataset = args.dataset
    comm_freq = args.comm_freq # how many epochs does a round of communication take place



    use_contrastive = args.use_contrastive
    # use_contrastive = True

    config = {
        # "algo": ["monolithic", "modular"],
        # "algo": "monolithic",
        "algo": "modular",
        "seed": seed,
        # "parallel": True,
        "parallel": False,
        "agent.batch_size": batch_size,
        # "num_agents": 8,
        # "num_agents": 2,
        "num_agents": 1,
        "dataset": "mnist",
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        # "dataset.num_tasks": num_tasks-num_init_tasks,  # NOTE: we already jointly
        "dataset.num_tasks": num_tasks,  # NOTE: we already jointly
        # train using a fake agent.
        "net": "mlp",
        "net.depth": num_init_tasks,
        "num_init_tasks": num_init_tasks,
        "net.dropout": 0.0,
        "train.num_epochs": args.num_epochs,
        "train.init_num_epochs": args.num_epochs,
        ## NOTE: TMP TO NOT UPDATE_MODULES
        "train.init_component_update_freq": args.num_epochs+1,
        "train.component_update_freq": args.num_epochs+1,
        "train.save_freq": save_freq,
        "agent.use_contrastive": use_contrastive,
        "agent.memory_size": 32,
        "dataset": dataset,
        "root_save_dir": f"experiment_results/debug_joint_agent_use_reg_fleet_comm_freq_{comm_freq}_use_contrastive_{use_contrastive}",
        "sharing_strategy": "debug_joint",
        "sharing_strategy.comm_freq": comm_freq,
        "sharing_strategy.when_reoptimize_structure": args.when_reoptimize_structure,
    }



    run_experiment(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
