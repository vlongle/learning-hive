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
parser.add_argument('--prefilter_strategy', type=str, default="oracle", choices=[
                    "oracle", "raw_distance", "none"], help='Pre-filtering strategy for the experiment.')
parser.add_argument('--scorer', type=str, default="cross_entropy", choices=[
    'cross_entropy', 'least_confidence', 'margin', 'entropy', 'random'], help='Scorer for the experiment.')
parser.add_argument('--add_data_prefilter_strategy', type=str, default="both", choices=[
    'task_neighbors_prefilter', 'global_y_prefilter', 'both'], help='Add data prefilter strategy for the experiment.')
# parser.add_argument('--assign_labels_strategy', type=str, default="same_as_query", choices=[
#     'groundtruth', 'same_as_query'], help='Assign labels strategy for the experiment.')
parser.add_argument('--assign_labels_strategy', type=str, default="groundtruth", choices=[
    'groundtruth', 'same_as_query'], help='Assign labels strategy for the experiment.')
parser.add_argument('--num_data_neighbors', type=int, default=5,
                    help='Number of data neighbors for the experiment.')
parser.add_argument('--num_queries', type=int, default=20,
                    help='Number of queries for the experiment.')
parser.add_argument('--num_comms_per_task', type=int, default=5,
                    help='Number of communications per task for the experiment.')

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
    # num_epochs = 10
    memory_size = 32
    if args.scale_shared_memory:
        shared_memory_size = max(args.num_queries * args.num_comms_per_task, memory_size)
    else:
        shared_memory_size = memory_size

    # shared_memory_size = memory_size
    # shared_memory_size = max(args.num_queries * args.num_comms_per_task, memory_size)

    query_task_mode = 'current' if args.algo == 'modular' else 'all'
    comm_freq = num_epochs // (args.num_comms_per_task + 1)

    num_agents = 8 if args.dataset != "combined" else 20
    sync_base = True if args.dataset == "combined" else False
    min_task = 0 if args.dataset != "combined" else 4

    if args.dataset != "cifar100":
        config = {
            "algo": args.algo,
            "agent.batch_size": batch_size,
            # "seed": args.seed,
            "seed": [0,1,2,3,4,5,6,7],
            "parallel": True,
            "num_agents": num_agents,
            "dataset": args.dataset,
            "sharing_strategy.min_task": min_task,
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
            # "root_save_dir": prefix + f"heuristic_experiment_results/recv_mem_{shared_memory_size}_freq_{comm_freq}",
            "root_save_dir": prefix + f"combined_recv_remove_neighbors_results/mem_size_{shared_memory_size}_comm_freq_{comm_freq}_num_queries_{args.num_queries}",



            "sharing_strategy": "recv_data",
            "sharing_strategy.remove_ood_neighbors": True,
            "sharing_strategy.shared_memory_size": shared_memory_size,
            "sharing_strategy.query_task_mode": query_task_mode,
            "sharing_strategy.num_data_neighbors": args.num_data_neighbors,
            "sharing_strategy.num_queries": args.num_queries,
            "sharing_strategy.comm_freq": comm_freq,
            "sharing_strategy.prefilter_strategy": args.prefilter_strategy,
            "sharing_strategy.add_data_prefilter_strategy": args.add_data_prefilter_strategy,
            "sharing_strategy.assign_labels_strategy": args.assign_labels_strategy,
            "sharing_strategy.scorer": args.scorer,
            "sharing_strategy.sync_base": sync_base,
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
            "dataset.num_tasks": 20,
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



            "root_save_dir": prefix + f"heuristic_experiment_results/recv_mem_{shared_memory_size}_freq_{comm_freq}",
            "sharing_strategy": "recv_data",
            "sharing_strategy.shared_memory_size": shared_memory_size,
            "sharing_strategy.query_task_mode": query_task_mode,
            "sharing_strategy.num_data_neighbors": args.num_data_neighbors,
            "sharing_strategy.num_queries": args.num_queries,
            "sharing_strategy.comm_freq": comm_freq,
            "sharing_strategy.prefilter_strategy": args.prefilter_strategy,
            "sharing_strategy.add_data_prefilter_strategy": args.add_data_prefilter_strategy,
            "sharing_strategy.assign_labels_strategy": args.assign_labels_strategy,
            "sharing_strategy.scorer": args.scorer,
            "sharing_strategy.sync_base": False,
        }


    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
