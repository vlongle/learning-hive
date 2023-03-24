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
        "algo": ["monolithic", "modular"],
        "seed": 0,
        "parallel": True,
        "num_agents": 4,
        "dataset": "mnist",
        "dataset.num_trains_per_class": [64, 128, -1],
        "dataset.num_vals_per_class": [50, 100, -1],
        "dataset.remap_labels": False,  # no remapping labels would absolutely wreck this
        "dataset.with_replacement": False,
        "dataset.num_tasks": 5,
        "net": "mlp",
        "net.depth": 2,
        "num_init_tasks": 2,
        "net.dropout": 0.0,
        "train.num_epochs": 200,
        "train.component_update_freq": 200,
        "root_save_dir": "vanilla_mnist_results",
        "agent.use_contrastive": [True, False],
        # "dataset": ["kmnist", "fashionmnist"],
    }
    run_experiment(config)

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
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
