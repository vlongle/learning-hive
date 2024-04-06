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
from shell.utils.utils import on_desktop


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
                    "mnist", "kmnist", "fashionmnist", "cifar100", "combined"], help='Dataset for the experiment.')
parser.add_argument('--sync_base', type=str2bool, default=True)
# parser.add_argument('--sync_base', type=str2bool, default=False)
parser.add_argument('--opt_with_random', type=str2bool, default=False)
parser.add_argument('--freeze_candidate_module', type=str2bool, default=False)
parser.add_argument('--transfer_decoder', type=str2bool, default=True)
parser.add_argument('--transfer_structure', type=str2bool, default=True)
parser.add_argument('--no_sparse_basis', type=str2bool, default=True)
# parser.add_argument('--num_tryout_epochs', type=int, default=100)
# parser.add_argument('--max_num_modules_tryout', type=int, default=14)
parser.add_argument('--num_tryout_epochs', type=int, default=20)
parser.add_argument('--max_num_modules_tryout', type=int, default=3)
parser.add_argument('--num_shared_module', type=int, default=1)
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
    num_init_tasks = 4
    num_tasks = 10
    batch_size = 64
    num_epochs = 100

    config = {

        "algo": "modular",
        "dataset": args.dataset,
        "agent.batch_size": batch_size,
        # "seed": args.seed,
        "seed": [0, 1, 2, 3, 4, 5, 6, 7],
        "topology": args.topology,
        "edge_drop_prob": args.edge_drop_prob,
        "parallel": True,
        "num_agents": 8,
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": num_tasks,
        "net": "mlp",
        "net.depth": num_init_tasks,
        'net.no_sparse_basis': args.no_sparse_basis,
        "num_init_tasks": num_init_tasks,
        "net.dropout": 0.5,
        "train.num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "train.init_num_epochs": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.save_freq": 10,
        "agent.use_contrastive": False,
        "agent.memory_size": 32,



        "sharing_strategy": "modmod",
        "sharing_strategy.comm_freq": num_epochs,  # once per task
        "sharing_strategy.opt_with_random": args.opt_with_random,
        "sharing_strategy.sync_base": args.sync_base,
        "sharing_strategy.freeze_candidate_module": args.freeze_candidate_module,
        "sharing_strategy.transfer_decoder": args.transfer_decoder,
        "sharing_strategy.transfer_structure": args.transfer_structure,



        "sharing_strategy.ranker": "label",
        # "sharing_strategy.module_select": "tryout",
        "sharing_strategy.module_select": "trust_sim",

        "sharing_strategy.num_shared_module": args.num_shared_module,
        "sharing_strategy.num_tryout_epochs": args.num_tryout_epochs,
        "sharing_strategy.max_num_modules_tryout": args.max_num_modules_tryout,
        # "root_save_dir": prefix + f"budget_experiment_results/modmod/tryout_epochs_{args.num_tryout_epochs}_max_modules_{args.max_num_modules_tryout}_num_shared_modules_{args.num_shared_module}_jorge_setting_lowest_task_id_wins_modmod_test_sync_base_{args.sync_base}_opt_with_random_{args.opt_with_random}_frozen_{args.freeze_candidate_module}_transfer_decoder_{args.transfer_decoder}_transfer_structure_{args.transfer_structure}_no_sparse_basis_{args.no_sparse_basis}",
        "root_save_dir": prefix + f"topology_experiment_results/modmod/topology_{args.topology}_edge_drop_{args.edge_drop_prob}",

        "overwrite": False,
    }

    # # # === CNN experiments: CIFAR100 ===

    # config = {
    #     "algo": "modular",
    #     # "seed": args.seed,
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7],
    #     "num_agents": 8,
    #     "parallel": True,
        
    #     "topology": args.topology,
    #     "edge_drop_prob": args.edge_drop_prob,

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
    #     'net.no_sparse_basis': args.no_sparse_basis,

    #     "sharing_strategy": "modmod",
    #     "sharing_strategy.comm_freq": num_epochs,  # once per task
    #     "sharing_strategy.opt_with_random": args.opt_with_random,
    #     "sharing_strategy.sync_base": args.sync_base,
    #     "sharing_strategy.freeze_candidate_module": args.freeze_candidate_module,
    #     "sharing_strategy.transfer_decoder": args.transfer_decoder,
    #     "sharing_strategy.transfer_structure": args.transfer_structure,

    #     # "sharing_strategy.ranker": "instance",
    #     "sharing_strategy.ranker": "label",
    #     "sharing_strategy.module_select": "tryout",
    #     "sharing_strategy.num_shared_module": args.num_shared_module,
    #     "sharing_strategy.num_tryout_epochs": args.num_tryout_epochs,
    #     "sharing_strategy.max_num_modules_tryout": args.max_num_modules_tryout,
    #     # "root_save_dir": prefix + f"budget_experiment_results/modmod/tryout_epochs_{args.num_tryout_epochs}_max_modules_{args.max_num_modules_tryout}_num_shared_modules_{args.num_shared_module}_jorge_setting_lowest_task_id_wins_modmod_test_sync_base_{args.sync_base}_opt_with_random_{args.opt_with_random}_frozen_{args.freeze_candidate_module}_transfer_decoder_{args.transfer_decoder}_transfer_structure_{args.transfer_structure}_no_sparse_basis_{args.no_sparse_basis}",
    #     "root_save_dir": prefix + f"topology_experiment_results/modmod/topology_{args.topology}_edge_drop_{args.edge_drop_prob}",
    # }

    # print('args', args, type(args.sync_base), type(args.opt_with_random))
    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
