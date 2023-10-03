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

parser = argparse.ArgumentParser(description='Run experiment with a specified seed.')
parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment.')
args = parser.parse_args()


if __name__ == "__main__":
    start = time.time()


    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===


    num_epochs = 20

    seed = args.seed
    config = {
        "algo": ["monolithic", "modular"],
        "seed": seed,
        "parallel": True,
        "num_agents": 8,
        "dataset": "mnist",
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": 10,
        "net": "mlp",
        "net.depth": 4,
        "num_init_tasks": 4,
        "net.dropout": 0.0,
        "train.num_epochs": num_epochs,
        "train.component_update_freq": num_epochs,
        "train.init_num_epochs": num_epochs,
        "train.init_component_update_freq": num_epochs,
        "train.save_freq": 20,
        "agent.use_contrastive": [True, False],
        "agent.memory_size": 32,
        "dataset": ["mnist", "kmnist", "fashionmnist"],
        "root_save_dir": "experiment_results/vanilla",
    }


    # # # === CNN experiments: CIFAR100 ===
    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     "algo": "modular",
    #     "seed": 0,
    #     "num_agents": 8,
    #     "parallel": True,
    #     "dataset": "cifar100",
    #     "dataset.num_trains_per_class": 256,
    #     "dataset.num_vals_per_class": -1,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": False,
    #     "dataset.num_tasks": 20,
    #     "net": "cnn",
    #     "net.depth": 4,
    #     "num_init_tasks": 4,
    #     "net.dropout": 0.5,
    #     "train.init_num_epochs": 1000,
    #     "train.init_component_update_freq": 1000,
    #     "train.num_epochs": 400,
    #     "train.component_update_freq": 400,
    #     "agent.memory_size": 32,
    #     # "agent.batch_size": 1024,
    #     "agent.batch_size": 128,
    #     "train.save_freq": 20,
    #     # "agent.use_contrastive": True,
    #     "agent.use_contrastive": False,
    #     # "root_save_dir": "cifar_task_specific_proj_allow_decoder_change_accommodate_train_500_epochs_temp_0.07_results",
    #     # "root_save_dir": "cifar_batch_size_128_epochs_400_double_temp_0.07_results",
    #     "root_save_dir": "cifar_baseline_double_epoch_results",
    # }

    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     # "algo": "modular",
    #     "algo": ["modular", "monolithic"],
    #     # "seed": [0, 1, 2, 3],
    #     "seed": 0,
    #     # "seed": 0,
    #     "num_agents": 4,
    #     "parallel": True,
    #     "dataset": "cifar100",
    #     "dataset.num_trains_per_class": 300,
    #     "dataset.num_vals_per_class": -1,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": False,
    #     "dataset.num_tasks": 20,
    #     "net": "cnn",
    #     "net.depth": 4,
    #     "num_init_tasks": 4,
    #     "net.dropout": 0.5,
    #     "train.init_num_epochs": 500,
    #     "train.init_component_update_freq": 500,
    #     "train.num_epochs": 500,
    #     "train.component_update_freq": 500,
    #     "agent.memory_size": 64,
    #     "agent.batch_size": 64,
    #     "train.save_freq": 20,
    #     # "train.save_freq": 1,
    #     # "agent.use_contrastive": [True, False],
    #     "agent.use_contrastive": False,
    #     # "root_save_dir": "cifar_lasttry_dropout_moreseed_epoch_200_results",
    #     # "root_save_dir": "cifar_no_updates_contrastive_results",
    #     "root_save_dir": "cifar_epochs_500_mild_dropout_memory_64_data_300_results",
    # }


    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")

