'''
File: /combine_modes_experiments.py
Project: experiments
Created Date: Tuesday April 23rd 2024
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2024 Long Le
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
parser.add_argument('--topology', type=str, default='fully_connected')
parser.add_argument('--edge_drop_prob', type=float, default=0.0)
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "modular", "monolithic"], help='Algorithm for the experiment.')

args = parser.parse_args()


if on_desktop():
    prefix = ""
else:
    prefix = "/mnt/kostas-graid/datasets/vlongle/"


"""
NEED TO BE VERY CAREFUL TO STUFF LIKE DYNAMICALLY SETTING CONFIG LIKE
QUERY_TASK_MODE = "CURRENT" IF ALGO == MODULAR ELSE ALL

"""
if __name__ == "__main__":
    start = time.time()

    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    num_init_tasks = 4
    num_tasks = 10
    batch_size = 64
    num_epochs = 100
    # num_epochs = 10
    num_agents = 20 if args.dataset == "combined" else 8

    combine = 'recv_data'
    # combine = 'modmod'
    sync_base = "grad" in combine or "modmod" in combine

    # NOTE: HACK: BUG::: min_task == 4 is going to be a problem
    # with the combined data recv in this config..., also same_as_query for MNIST variants,
    # and also somehow no_sparse_basis=False??

    no_sparse_basis = False  # the same level
    # no_sparse_basis = True  # this is actually much worse

    root_save_dir = prefix + \
        f"combine_modes_results/{combine}_no_sparse_{no_sparse_basis}"
    if args.dataset != "cifar100":
        config = {

            "algo": "modular",
            "dataset": args.dataset,
            "seed": args.seed,
            # "seed": [0, 1, 2, 3, 4, 5, 6, 7],
            "num_agents": num_agents,
            "agent.batch_size": batch_size,
            "topology": args.topology,
            "edge_drop_prob": args.edge_drop_prob,
            "parallel": True,
            "dataset.num_trains_per_class": 64,
            "dataset.num_vals_per_class": 50,
            "dataset.remap_labels": True,
            "dataset.with_replacement": True,
            "dataset.num_tasks": num_tasks,
            "net": "mlp",
            "net.depth": num_init_tasks,
            'net.no_sparse_basis': no_sparse_basis,
            "num_init_tasks": num_init_tasks,
            "net.dropout": 0.5,
            "train.num_epochs": num_epochs,
            "train.component_update_freq": num_epochs,
            "train.init_num_epochs": num_epochs,
            "train.init_component_update_freq": num_epochs,
            "train.save_freq": 10,
            "agent.use_contrastive": False,
            "agent.memory_size": 32,


            "sharing_strategy": "combine_modes",
            # "sharing_strategy.communicator": "'modmod,grad_sharing_prox,recv_data'",
            "sharing_strategy.communicator": combine,
            "sharing_strategy.sync_base": sync_base,
            "root_save_dir": root_save_dir,

        }

    else:
        config = {
            "algo": "modular",
            "seed": args.seed,
            # "seed": [0, 1, 2, 3, 4, 5, 6, 7],
            "num_agents": num_agents,
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


            "root_save_dir": root_save_dir,


            "sharing_strategy": "combine_modes",
            # "sharing_strategy.communicator": "'modmod,grad_sharing_prox,recv_data'",
            "sharing_strategy.communicator": combine,
            "sharing_strategy.sync_base": sync_base,
            "root_save_dir": root_save_dir,
        }

    # print('args', args, type(args.sync_base), type(args.opt_with_random))
    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
