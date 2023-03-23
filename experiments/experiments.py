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
if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    config = {
        # "algo": ["monolithic", "modular"],
        # "algo": "monolithic",
        "algo": "modular",
        "seed": 0,
        "parallel": False,
        # "num_agents": 4,
        "num_agents": 1,
        # "dataset": ["mnist", "kmnist", "fashionmnist"],
        "dataset": "mnist",
        # "dataset.num_trains_per_class": 64,
        "dataset.num_trains_per_class": -1,
        # "dataset.num_vals_per_class": 50,
        "dataset.num_vals_per_class": -1,
        # "dataset.remap_labels": False,
        "dataset.remap_labels": False,  # no remapping labels would absolutely wreck this
        # bitch's algorithm! with num_init_tasks=4
        "dataset.with_replacement": False,
        "dataset.num_tasks": 5,
        # "dataset.num_tasks": 4,
        # "agent.batch_size": 32,
        "net": "mlp",
        "net.depth": 2,
        "num_init_tasks": 2,
        "net.dropout": 0.0,
        # "net.dropout": 0.5,
        "net.freeze_encoder": True,
        # "net.freeze_encoder": False,
        "train.num_epochs": 20,
        "train.component_update_freq": 20,
        # "train.num_epochs": 1,
        # "train.component_update_freq": 1,
        # "root_save_dir": "testing_contrastive_results_20epochs",
        # "root_save_dir": "testing_contrastive_results",
        "root_save_dir": "testing_contrastive_wo_replacement_results",
    }
    run_experiment(config)

    # config = {
    #     "algo": ["monolithic", "modular"],
    #     "seed": 0,
    #     "num_agents": 4,
    #     # "dataset": ["mnist", "kmnist", "fashionmnist"],
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": -1,
    #     "dataset.num_vals_per_class": -1,
    #     "dataset.remap_labels": False,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": 10,
    #     "net": "mlp",
    #     "net.depth": 4,
    #     "net.num_init_tasks": 4,
    #     "net.dropout": 0.5,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "root_save_dir": "full_data_results",
    # }
    # run_experiment(config)

    # # # === CNN experiments: CIFAR100 ===
    # config = {
    #     "algo": ["monolithic", "modular"],
    #     "seed": 0,
    #     "num_agents": 4,
    #     "dataset": "cifar100",
    #     "dataset.num_trains_per_class": 256,
    #     "dataset.num_vals_per_class": -1,
    #     "dataset.num_tasks": 20,
    #     "net": "cnn",
    #     "net.num_init_tasks": 4,
    #     "net.dropout": 0.5,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "root_save_dir": "results",
    # }

    # run_experiment(config)

    # config = {
    #     "algo": ["monolithic", "modular"],
    #     "seed": 0,
    #     "num_agents": 4,
    #     "dataset": "cifar100",
    #     "dataset.num_trains_per_class": -1,  # <<< only change: use all training data
    #     "dataset.num_vals_per_class": -1,
    #     "dataset.num_tasks": 20,
    #     "net": "cnn",
    #     "net.num_init_tasks": 4,
    #     "net.dropout": 0.5,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "root_save_dir": "full_data_results",
    # }

    # run_experiment(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
