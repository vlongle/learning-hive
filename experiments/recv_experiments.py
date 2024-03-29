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
parser = argparse.ArgumentParser(
    description='Run experiment with a specified seed.')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for the experiment.')
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "modular", "monolithic"], help='Algorithm for the experiment.')
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
parser.add_argument('--prefilter_strategy', type=str, default="oracle", choices=[
                    "oracle", "raw_distance", "none"], help='Pre-filtering strategy for the experiment.')
parser.add_argument('--scorer', type=str, default="cross_entropy", choices=[
    'cross_entropy', 'least_confidence', 'margin', 'entropy', 'random'], help='Scorer for the experiment.')
parser.add_argument('--add_data_prefilter_strategy', type=str, default="both", choices=[
    'task_neighbors_prefilter', 'global_y_prefilter', 'both'], help='Add data prefilter strategy for the experiment.')
parser.add_argument('--assign_labels_strategy', type=str, default="same_as_query", choices=[
    'groundtruth', 'same_as_query'], help='Assign labels strategy for the experiment.')
parser.add_argument('--num_data_neighbors', type=int, default=5,
                    help='Number of data neighbors for the experiment.')
parser.add_argument('--num_queries', type=int, default=20,
                    help='Number of queries for the experiment.')
parser.add_argument('--num_comms_per_task', type=int, default=5,
                    help='Number of communications per task for the experiment.')
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

    config = {
        "algo": args.algo,
        "agent.batch_size": batch_size,
        # "seed": args.seed,
        "seed": [0, 1, 2, 3, 4, 5, 6, 7],
        "parallel": True,
        "num_agents": 8,
        "dataset": "mnist",
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
        "dataset": args.dataset,
        # "agent.use_ood_separation_loss": False,
        "root_save_dir": prefix + f"experiment_results/jorge_setting_recv",
        "sharing_strategy": "recv_data",
        "sharing_strategy.shared_memory_size": memory_size,
        "sharing_strategy.query_task_mode": query_task_mode,
        "sharing_strategy.num_data_neighbors": args.num_data_neighbors,
        "sharing_strategy.num_queries": args.num_queries,
        "sharing_strategy.comm_freq": comm_freq,
        "sharing_strategy.prefilter_strategy": args.prefilter_strategy,
        "sharing_strategy.add_data_prefilter_strategy": args.add_data_prefilter_strategy,
        "sharing_strategy.assign_labels_strategy": args.assign_labels_strategy,
        "sharing_strategy.scorer": args.scorer,
    }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
