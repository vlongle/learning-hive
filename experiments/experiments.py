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
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
parser.add_argument('--no_sparse_basis', type=str2bool, default=False)
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "monolithic", "modular"], help='Algorithm for the experiment.')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--memory_size', type=int, default=32)
parser.add_argument('--num_trains_per_class', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


if on_desktop():
    prefix = ""
else:
    prefix = "/mnt/kostas-graid/datasets/vlongle/"


if __name__ == "__main__":
    start = time.time()

    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    num_epochs = 100
    # num_epochs = 2
    # num_epochs = 5
    num_init_tasks = 4
    # num_tasks = 10
    batch_size = 64

    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     # "algo": "modular",
    #     "algo": "monolithic",
    #     "agent.batch_size": batch_size,
    #     # "seed": [0, 1, 2, 3, 4, 5, 6, 7],
    #     "seed": args.seed,
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
    #     "num_init_tasks": num_init_tasks,
    #     # "net.dropout": 0.0,
    #     "net.dropout": 0.5,
    #     "train.num_epochs": num_epochs,
    #     "train.component_update_freq": num_epochs,
    #     "train.init_num_epochs": num_epochs,
    #     "train.init_component_update_freq": num_epochs,
    #     "net.no_sparse_basis": args.no_sparse_basis,
    #     "train.save_freq": 10,
    #     # "agent.use_contrastive": [True, False],
    #     "agent.use_contrastive": True,
    #     # "agent.use_contrastive": False,
    #     "agent.memory_size": 32,
    #     "dataset": args.dataset,  # use the dataset from arguments
    #     # "dataset": ["mnist", "kmnist", "fashionmnist"],
    #     # "root_save_dir": f"experiment_results/debug",
    #     "root_save_dir": f"experiment_results/vanilla_jorge_setting_no_sparse",
    # }

    # # # === CNN experiments: CIFAR100 ===
    # more stuff to try to recover the prev performance
    # decrease dropout to 0.0, no_sparse_basis=False, memory_size
    config = {
        "algo": args.algo,
        "seed": args.seed,
        # "num_agents": 4,
        "num_agents": 1,
        "parallel": True,
        # "parallel": False,
        "dataset": "cifar100",
        "dataset.num_trains_per_class": args.num_trains_per_class,
        "dataset.num_vals_per_class": -1,
        "dataset.remap_labels": True,
        "dataset.with_replacement": False,
        "net": "cnn",
        "net.depth": 4,
        "num_init_tasks": 4,
        "dataset.num_tasks": 20,
        # "dataset.num_tasks": 4,
        # "net.dropout": 0.0,
        "net.dropout": args.dropout,
        "train.init_num_epochs": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "agent.memory_size": args.memory_size,
        # "agent.batch_size": 1024,
        # "agent.batch_size": args.batch_size,
        "agent.batch_size": 1000,
        "train.save_freq": 10,
        "agent.use_contrastive": False,
        "net.no_sparse_basis": args.no_sparse_basis,
        "root_save_dir": prefix + f"experiment_results/debug_cifar100_vanilla_jorge_setting_dropout_{args.dropout}_memory_{args.memory_size}_no_sparse_{args.no_sparse_basis}_num_trains_{args.num_trains_per_class}_batchsize_{args.batch_size}",
    }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
