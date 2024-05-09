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
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "modular", "monolithic"], help='Algorithm for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100", "combined"], help='Dataset for the experiment.')
parser.add_argument('--num_comms_per_task', type=int, default=5,
                    help='Number of communications per task for the experiment.')
parser.add_argument('--budget', type=int, default=20,
                    help='Budget for the experiment.')
parser.add_argument('--enforce_balance', type=str2bool, default=True,
                    help='Enforce balance for the experiment.')
parser.add_argument('--hash_data', type=str2bool, default=True,
                    help='Whether to dedup shared data.')
parser.add_argument('--scale_shared_memory', type=str2bool, default=True)

args = parser.parse_args()

if on_desktop():
    prefix = ""
else:
    prefix = "/mnt/kostas-graid/datasets/vlongle/learning_hive/"

if __name__ == "__main__":
    start = time.time()

    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    num_init_tasks = 4
    num_tasks = 10
    batch_size = 64
    num_epochs = 100
    memory_size = 32

    query_task_mode = 'current' if args.algo == 'modular' else 'all'
    comm_freq = num_epochs // (args.num_comms_per_task + 1)
    num_agents = 20 if args.dataset == "combined" else 8
    sync_base = True if args.dataset == "combined" else False

    if args.scale_shared_memory:
        shared_memory_size = max(args.budget * num_agents, memory_size)
    else:
        shared_memory_size = memory_size

    min_task = 4 if args.dataset == "combined" else 0

    root_save_dir = f"debug_combine_modes_results/gt_recv_data_no_sparse_False_recv_mod_add_data_backward_True_make_new_opt_True"

    if args.dataset != "cifar100":
        config = {
            "algo": args.algo,
            "agent.batch_size": batch_size,
            "seed": args.seed,
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

            "root_save_dir": prefix + f"heuristic_experiment_results/heuristic_budget_{args.budget}_enforce_balance_{args.enforce_balance}_mem_{shared_memory_size}_freq_{comm_freq}",
            "sharing_strategy": "heuristic_data",
            "sharing_strategy.shared_memory_size": shared_memory_size,
            "sharing_strategy.comm_freq": comm_freq,
            "sharing_strategy.sync_base": sync_base,
            "sharing_strategy.query_task_mode": query_task_mode,
            "sharing_strategy.budget": args.budget,
            "sharing_strategy.enforce_balance": args.enforce_balance,
            "sharing_strategy.min_task": min_task,
        }
    else:
        config = {

            "algo": args.algo,
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
            "dataset.num_tasks": 5,
            "net.dropout": 0.5,
            "train.init_num_epochs": num_epochs,
            "train.init_component_update_freq": num_epochs,
            "train.num_epochs": num_epochs,
            "train.component_update_freq": num_epochs,
            "agent.memory_size": memory_size,
            "agent.batch_size": batch_size,
            "train.save_freq": 10,
            "agent.use_contrastive": False,
            "net.no_sparse_basis": True,



            "root_save_dir": root_save_dir,
            "sharing_strategy": "heuristic_data",
            "sharing_strategy.shared_memory_size": shared_memory_size,
            "sharing_strategy.comm_freq": comm_freq,
            "sharing_strategy.sync_base": False,
            "sharing_strategy.query_task_mode": query_task_mode,
            "sharing_strategy.budget": args.budget,
            "sharing_strategy.enforce_balance": args.enforce_balance,
            "sharing_strategy.hash_data": args.hash_data,
        }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
