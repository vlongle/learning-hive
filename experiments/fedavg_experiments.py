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
args = parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===

    # ACTUAL CONFIG
    num_init_tasks = 4
    num_tasks = 10
    # num_tasks = 5
    num_epochs = 100
    # comm_freq = 1
    # comm_freq = 50 # how many epochs does a round of communication take place

    batch_size = 64
    save_freq = 1

    seed = args.seed
    dataset = args.dataset
    comm_freq = args.comm_freq  # how many epochs does a round of communication take place

    config = {
        # "algo": ["monolithic", "modular"],
        "algo": "monolithic",
        "seed": args.seed,
        # "seed": [0, 1, 2, 3],
        "parallel": True,
        # "parallel": False,
        # "num_agents": 8,
        "num_agents": 4,
        "dataset": args.dataset,
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
        "root_save_dir": "experiment_results/jorge_setting_fedavg",
        # ================================================
        # GRAD SHARING SETUP
        "sharing_strategy": "grad_sharing",
        "sharing_strategy.num_coms_per_round": 1,
        "sharing_strategy.comm_freq": 5,
        # "sharing_strategy.log_freq": 10,

        # ================================================
    }

    # config = {
    #     # "algo": "modular",
    #     "algo": "monolithic",
    #     "seed": 0,
    #     # "parallel": True,
    #     "parallel": False,
    #     "num_agents": 2,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     # "dataset.num_tasks": num_tasks-num_init_tasks,  # NOTE: we already jointly
    #     "dataset.num_tasks": num_tasks,  # NOTE: we already jointly
    #     # train using a fake agent.
    #     "net": "mlp",
    #     "net.depth": num_init_tasks,
    #     "num_init_tasks": num_init_tasks,
    #     "net.dropout": 0.0,
    #     "train.num_epochs": num_epochs,
    #     "train.component_update_freq": num_epochs,
    #     "train.init_num_epochs": num_epochs,
    #     "train.init_component_update_freq": num_epochs,
    #     "train.save_freq": 20,
    #     "agent.use_contrastive": True,
    #     "agent.memory_size": 32,
    #     "dataset": "mnist",
    #     "root_save_dir": "experiment_results/fl/",
    #     "sharing_strategy": "grad_sharing",
    #     "sharing_strategy.comm_freq": comm_freq,
    # }

    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     # "algo": "monolithic",
    #     "algo": "modular",
    #     "seed": seed,
    #     "parallel": True,
    #     # "parallel": False,
    #     "agent.batch_size": batch_size,
    #     "num_agents": 8,
    #     # "num_agents": 2,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     # "dataset.num_tasks": num_tasks-num_init_tasks,  # NOTE: we already jointly
    #     "dataset.num_tasks": num_tasks,  # NOTE: we already jointly
    #     # train using a fake agent.
    #     "net": "mlp",
    #     "net.depth": num_init_tasks,
    #     "num_init_tasks": num_init_tasks,
    #     "net.dropout": 0.0,
    #     "train.num_epochs": args.num_epochs,
    #     "train.component_update_freq": args.num_epochs,
    #     "train.init_num_epochs": args.num_epochs,
    #     "train.init_component_update_freq": args.num_epochs,
    #     "train.save_freq": save_freq,
    #     "agent.use_contrastive": True,
    #     "agent.memory_size": 32,
    #     "dataset": dataset,
    #     "root_save_dir": f"debug_fix_experiment_results/fedavg_epochs_{args.num_epochs}_freq_{comm_freq}_only_final_{args.when_reoptimize_structure}/",
    #     "sharing_strategy": "grad_sharing",
    #     "sharing_strategy.comm_freq": comm_freq,
    #     "sharing_strategy.when_reoptimize_structure": args.when_reoptimize_structure,
    # }

    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     "algo": "monolithic",
    #     "seed": 0,
    #     "parallel": True,
    #     "num_agents": 8,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": 10-num_init_tasks,  # NOTE: we already jointly
    #     # train using a fake agent.
    #     "net": "mlp",
    #     "net.depth": 4,
    #     "num_init_tasks": num_init_tasks,
    #     "net.dropout": 0.0,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "train.init_num_epochs": 100,
    #     "train.init_component_update_freq": 100,
    #     "train.save_freq": 1,
    #     "agent.use_contrastive": True,
    #     "agent.memory_size": 32,
    #     # "dataset": ["mnist", "kmnist", "fashionmnist"],
    #     "dataset": "mnist",
    #     # "root_save_dir": "grad_new_unfreeze_all_decoders_retrain_results",
    #     "root_save_dir": "grad_more_log_debug_results",
    #     # ================================================
    #     # GRAD SHARING SETUP
    #     "sharing_strategy": "grad_sharing",
    #     "sharing_strategy.num_coms_per_round": 50,
    #     "sharing_strategy.retrain.num_epochs": 5,
    #     "sharing_strategy.log_freq": 1,

    #     # ================================================
    # }

    # TOY CONFIG
    # num_init_tasks = 4
    # config = {
    #     "algo": "modular",
    #     "seed": 0,
    #     # "parallel": True,
    #     "parallel": False,
    #     "num_agents": 2,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": 10-num_init_tasks,  # NOTE: we already jointly
    #     # train using a fake agent.
    #     "net": "mlp",
    #     "net.depth": 4,
    #     "num_init_tasks": num_init_tasks,
    #     "net.dropout": 0.0,

    #     "train.num_epochs": 1,
    #     "train.component_update_freq": 1,
    #     "train.init_num_epochs": 1,
    #     "train.init_component_update_freq": 1,

    #     "train.save_freq": 20,
    #     "agent.use_contrastive": True,
    #     "agent.memory_size": 32,
    #     "dataset": "mnist",
    #     "root_save_dir": "test_grad_results",
    #     # ================================================
    #     # GRAD SHARING SETUP
    #     "sharing_strategy": "grad_sharing",
    #     "sharing_strategy.num_coms_per_round": 5,
    #     "sharing_strategy.retrain.num_epochs": 1,
    #     # ================================================
    # }

    run_experiment(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
