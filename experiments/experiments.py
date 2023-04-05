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
    #     "root_save_dir": "vanilla_results",
    #     "agent.use_contrastive": [True, False],
    #     "agent.memory_size": 32,
    #     "dataset": ["mnist", "kmnist", "fashionmnist"],
    # }

    # run_experiment(config)
    # 1 hour for around 100 epochs, contrastive=False, ~ 76%

    # # # === CNN experiments: CIFAR100 ===
    config = {
        # "algo": ["monolithic", "modular"],
        "algo": "modular",
        "seed": 1,
        "num_agents": 4,
        "parallel": True,
        "dataset": "cifar100",
        "dataset.num_trains_per_class": 256,
        "dataset.num_vals_per_class": -1,
        "dataset.remap_labels": True,
        "dataset.with_replacement": False,
        "dataset.num_tasks": 20,
        "net": "cnn",
        "net.depth": 4,
        "num_init_tasks": 4,
        "net.dropout": 0.5,
        "train.init_num_epochs": 500,
        "train.init_component_update_freq": 500,
        "train.num_epochs": 100,
        "train.component_update_freq": 100,
        "agent.memory_size": 32,
        # "agent.batch_size": 1024,
        "agent.batch_size": 64,
        "train.save_freq": 20,
        "agent.use_contrastive": True,
        # "agent.use_contrastive": [True, False],
        # "agent.use_contrastive": True,
        # "root_save_dir": "vanilla_results_wo_replacement",
        # "root_save_dir": "finding_hyper_for_mod_contrastive_large_lower_temp_results",
        # "root_save_dir": "finding_hyper_for_mod_contrastive_large_no_projector_results",
        # "root_save_dir": "finding_hyper_for_mod_contrastive_large_deeper_projector_results",
        # "root_save_dir": "finding_hyper_for_mod_contrastive_large_deeper_projector_freeze_proj_results",
        # "root_save_dir": "finding_hyper_for_mod_contrastive_large_deeper_projector_freeze_proj_more_epochs_results",
        "root_save_dir": "finding_hyper_for_mod_contrastive_large_deeper_projector_freeze_proj_more_epochs_less_dim_results",
    }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
