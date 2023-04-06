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

    num_init_tasks = 4
    # num_init_tasks = 2
    config = {
        "algo": "monolithic",
        "seed": 0,
        "parallel": True,
        # "parallel": False,
        "num_agents": 8,
        # "num_agents": 2,
        "dataset": "mnist",
        "dataset.num_trains_per_class": 64,
        "dataset.num_vals_per_class": 50,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": 10-num_init_tasks,  # NOTE: we already jointly
        # train using a fake agent.
        "net": "mlp",
        "net.depth": 4,
        "num_init_tasks": num_init_tasks,
        "net.dropout": 0.0,
        "train.num_epochs": 100,
        "train.component_update_freq": 100,
        "train.init_num_epochs": 100,
        "train.init_component_update_freq": 100,

        # "train.num_epochs": 1,
        # "train.component_update_freq": 1,
        # "train.init_num_epochs": 1,
        # "train.init_component_update_freq": 1,

        "train.save_freq": 20,
        "agent.use_contrastive": True,
        "agent.memory_size": 32,
        "dataset": "mnist",
        # "root_save_dir": "test_grad_results",
        "root_save_dir": "grad_results",
        # ================================================
        # GRAD SHARING SETUP
        "sharing_strategy": "grad_sharing",
        "sharing_strategy.num_coms_per_round": 50,
        "sharing_strategy.retrain.num_epochs": 5,
        "sharing_strategy.log_freq": 10,
        # "sharing_strategy.num_coms_per_round": 5,
        # "sharing_strategy.retrain.num_epochs": 1,
        # ================================================
    }

    run_experiment(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
