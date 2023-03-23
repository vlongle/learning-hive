'''
File: /grad_experiments.py
Project: experiments
Created Date: Tuesday March 21st 2023
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
        "algo": "monolithic",
        "seed": 0,
        "num_agents": 4,
        "dataset": "mnist",
        "parallel": False,
        # ================================================
        # GRAD SHARING SETUP
        "sharing_strategy": "grad_sharing",
        "sharing_strategy.num_coms_per_round": 50,
        "sharing_strategy.retrain.num_epochs": 1,
        # ================================================
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": False,
        "dataset.with_replacement": True,
        # "dataset.num_tasks": 10,
        "dataset.num_tasks": 4,
        "net": "mlp",
        "net.depth": 4,
        "net.num_init_tasks": 4,
        "net.dropout": 0.5,
        "net.freeze_encoder": True,
        "train.num_epochs": 50,
        "train.component_update_freq": 50,
        # "train.num_epochs": 5,
        # "train.component_update_freq": 5,
        "root_save_dir": "grad_results",
    }
    run_experiment(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
