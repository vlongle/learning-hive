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
    # config = {
    #     "algo": ["monolithic", "modular"],
    #     # "seed": [0, 1, 2, 3],
    #     "seed": 0,
    #     "parallel": True,
    #     "num_agents": 8,
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": 10,
    #     "net": "mlp",
    #     "net.depth": 4,
    #     "num_init_tasks": 4,
    #     "net.dropout": 0.0,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "train.save_freq": 20,
    #     "root_save_dir": "recv_results",
    #     "agent.use_contrastive": True,
    #     "agent.memory_size": 32,
    #     "sharing_strategy": "recv_data",
    #     "dataset": ["mnist", "kmnist", "fashionmnist"],
    # }

    # small debug experiment
    config = {
        "algo": ["monolithic"],
        "seed": [0, 1, 2, 3],
        # "seed": 0,
        "parallel": True,
        "num_agents": 2,
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
        "train.num_epochs": 100,
        "train.init_num_epochs": 100,
        "train.component_update_freq": 100,
        "train.save_freq": 1,
        "root_save_dir": "recv_results",
        "agent.use_contrastive": True,
        "agent.memory_size": 32,
        "sharing_strategy": "recv_data",
        "dataset": ["mnist", "kmnist", "fashionmnist"],
    }
    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
