'''
File: /experiments.py
Project: experiments
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import time
import datetime
from shell.utils.experiment_utils import run_experiment
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='Run experiment with a specified seed.')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
parser.add_argument('--sync_base', type=str2bool, default=True)
parser.add_argument('--opt_with_random', type=str2bool, default=False)
parser.add_argument('--freeze_candidate_module', type=str2bool, default=False)
parser.add_argument('--transfer_decoder', type=str2bool, default=False)
parser.add_argument('--transfer_structure', type=str2bool, default=False)
parser.add_argument('--no_sparse_basis', type=str2bool, default=False)
args = parser.parse_args()


if __name__ == "__main__":
    start = time.time()

    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    num_init_tasks = 4
    num_tasks = 10
    batch_size = 64
    num_epochs = 100
    # num_epochs = 5

    # config = {
    #     "algo": "modular",
    #     "agent.batch_size": batch_size,
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7],
    #     # "seed": [0, 1],
    #     # "seed": 0,
    #     "parallel": True,
    #     "num_agents": 8,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": num_tasks,
    #     "net": "mlp",
    #     "net.depth": num_init_tasks,
    #     'net.no_sparse_basis': args.no_sparse_basis,
    #     "num_init_tasks": num_init_tasks,
    #     # "net.dropout": 0.0,
    #     "net.dropout": 0.5,
    #     "train.num_epochs": num_epochs,
    #     "train.component_update_freq": num_epochs,
    #     "train.init_num_epochs": num_epochs,
    #     "train.init_component_update_freq": num_epochs,
    #     "train.save_freq": 10,
    #     "agent.use_contrastive": [True, False],
    #     # "agent.use_contrastive": False,
    #     "agent.memory_size": 32,
    #     # "dataset": ["mnist", "kmnist", "fashionmnist"],
    #     "dataset": args.dataset,
    #     "sharing_strategy": "modmod",
    #     "sharing_strategy.comm_freq": num_epochs,  # once per task
    #     "sharing_strategy.opt_with_random": args.opt_with_random,
    #     "sharing_strategy.sync_base": args.sync_base,
    #     "sharing_strategy.freeze_candidate_module": args.freeze_candidate_module,
    #     "sharing_strategy.transfer_decoder": args.transfer_decoder,
    #     "sharing_strategy.transfer_structure": args.transfer_structure,
    #     "root_save_dir": f"experiment_results/jorge_setting_lowest_task_id_wins_modmod_test_sync_base_{args.sync_base}_opt_with_random_{args.opt_with_random}_frozen_{args.freeze_candidate_module}_transfer_decoder_{args.transfer_decoder}_transfer_structure_{args.transfer_structure}_no_sparse_basis_{args.no_sparse_basis}",
    #     "overwrite": False,
    # }

    # # # === CNN experiments: CIFAR100 ===

    config = {
        "algo": "modular",
        "seed": args.seed,
        "num_agents": 8,
        "parallel": True,
        "dataset": "cifar100",
        "dataset.num_trains_per_class": 256,
        "dataset.num_vals_per_class": -1,
        "dataset.remap_labels": True,
        "dataset.with_replacement": False,
        "net": "cnn",
        "net.depth": 4,
        "num_init_tasks": 4,
        "dataset.num_tasks": 20,
        # "net.dropout": 0.0,
        "net.dropout": 0.5,
        "train.init_num_epochs": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "agent.memory_size": 32,
        "agent.batch_size": 64,
        "train.save_freq": 10,
        "agent.use_contrastive": False,
        'net.no_sparse_basis': args.no_sparse_basis,

        "sharing_strategy": "modmod",
        "sharing_strategy.comm_freq": num_epochs,  # once per task
        "sharing_strategy.opt_with_random": args.opt_with_random,
        "sharing_strategy.sync_base": args.sync_base,
        "sharing_strategy.freeze_candidate_module": args.freeze_candidate_module,
        "sharing_strategy.transfer_decoder": args.transfer_decoder,
        "sharing_strategy.transfer_structure": args.transfer_structure,
        "root_save_dir": f"/mnt/kostas-graid/datasets/vlongle/learning_hive/experiment_results/jorge_setting_lowest_task_id_wins_modmod_test_sync_base_{args.sync_base}_opt_with_random_{args.opt_with_random}_frozen_{args.freeze_candidate_module}_transfer_decoder_{args.transfer_decoder}_transfer_structure_{args.transfer_structure}_no_sparse_basis_{args.no_sparse_basis}",
    }

    # print('args', args, type(args.sync_base), type(args.opt_with_random))
    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
