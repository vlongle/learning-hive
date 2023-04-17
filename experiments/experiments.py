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
    #     "seed": [1,2,3],
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
    #     "train.init_num_epochs": 100,
    #     "train.init_component_update_freq": 100,
    #     "train.save_freq": 20,
    #     "agent.use_contrastive": [True, False],
    #     "agent.memory_size": 32,
    #     "dataset": ["mnist", "kmnist", "fashionmnist"],
    #     "root_save_dir": "vanilla_more_seeds_results",
    # }

    # # # === CNN experiments: CIFAR100 ===
    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     "algo": "modular",
    #     "seed": 0,
    #     "num_agents": 2,
    #     # "parallel": True,
    #     "parallel": False,
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
    #     # "train.init_num_epochs": 500,
    #     # "train.init_component_update_freq": 500,
    #     "train.init_num_epochs": 5,
    #     "train.init_component_update_freq": 5,
    #     # "train.num_epochs": 200,
    #     # "train.component_update_freq": 200,
    #     "train.num_epochs": 2,
    #     "train.component_update_freq": 2,
    #     "agent.memory_size": 32,
    #     # "agent.batch_size": 1024,
    #     "agent.batch_size": 64,
    #     "train.save_freq": 20,
    #     "agent.use_contrastive": True,
    #     "root_save_dir": "cifar_task_specific_proj_results",
    # }

    config = {
        # "algo": ["monolithic", "modular"],
        "algo": "modular",
        "seed": 0,
        "num_agents": 1,
        # "parallel": True,
        "parallel": False,
        "dataset": "cifar100",
        # "dataset": "mnist",
        "dataset.num_trains_per_class": 256,
        "dataset.num_vals_per_class": -1,
        "dataset.remap_labels": True,
        "dataset.with_replacement": False,
        # "dataset.num_tasks": 20,
        "dataset.num_tasks": 5,
        "net": "cnn",
        # "net": "mlp",
        "net.depth": 4,
        "num_init_tasks": 4,
        "net.dropout": 0.5,
        # "net.dropout": 0.25,
        # "net.dropout": 0.0,
        # "train.init_num_epochs": 1000,
        # "train.init_component_update_freq": 1000,
        # "train.num_epochs": 400,
        # "train.component_update_freq": 400,
        # "train.init_num_epochs": 100,
        # "train.init_component_update_freq": 10,
        # "train.init_num_epochs": 100,

        # "train.num_epochs": 2,
        # "train.component_update_freq": 2,
        # "train.init_num_epochs": 1,
        # "train.init_component_update_freq": 1,


        "train.num_epochs": 100,
        "train.component_update_freq": 200,
        "train.init_num_epochs": 100,
        "train.init_component_update_freq": 200,

        # "train.init_num_epochs": 1,
        # "train.init_component_update_freq": 1,
        # "train.num_epochs": 2,
        # "train.component_update_freq": 2,
        "agent.memory_size": 32,
        # "agent.batch_size": 1024,
        # "agent.batch_size": 128,
        "agent.batch_size": 64,
        # "train.save_freq": 20,
        "train.save_freq": 1,
        # "agent.use_contrastive": True,
        "agent.use_contrastive": False,
        # "root_save_dir": "cifar_task_specific_proj_allow_decoder_change_accommodate_train_500_epochs_temp_0.07_results",
        # "root_save_dir": "cifar_batch_size_128_epochs_400_double_temp_0.07_results",
        # "root_save_dir": "cifar_debug_dropout_results",
        # "root_save_dir": "tmp_dropout_results",
        # "root_save_dir": "tmp_no_dropout_keep_transform_results",
        "root_save_dir": "tmp_no_dropout_no_update_modules_checkpt_results",
        # "root_save_dir": "tmp_play_results",
    }

    run_experiment(config, strict=False)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
