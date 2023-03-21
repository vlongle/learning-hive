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
import subprocess
import datetime
import os
def get_all_combinations(config):
    """
    Config is a dictionary where keys are string,
    and values are either a single value or a list of values.
    Return a list of dictionaries, where in each dictionary,
    a key is a string and a value is a single value by combining
    all possible values from config.
    """
    keys = list(config.keys())
    values = list(config.values())
    combs = []
    for i in range(len(values)):
        if type(values[i]) is not list:
            values[i] = [values[i]]
    for i in range(len(values[0])):
        combs.append({keys[0]: values[0][i]})
    for i in range(1, len(values)):
        new_combs = []
        for j in range(len(combs)):
            for k in range(len(values[i])):
                new_combs.append(combs[j].copy())
                new_combs[-1][keys[i]] = values[i][k]
        combs = new_combs
    return combs


def main(config):
    script_path = os.path.join("experiments", "run.py")

    combs = get_all_combinations(config)
    print(len(combs))

    for cfg in combs:
        dataset = cfg["dataset"]
        algo = cfg["algo"]
        cmd = [
            "python",
            script_path,
        ] + [f"{k}={v}" for k, v in cfg.items()]
        job_name = f"{dataset}_{algo}"
        cmd += [f"train={algo}", f"job_name={job_name}"]
        subprocess.run(cmd)


if __name__ == "__main__":
    start = time.time()
    # === MLP experiments: MNIST, KMNIST, FashionMNIST ===
    # config = {
    #     # "algo": ["monolithic", "modular"],
    #     "algo": "modular",
    #     "seed": 0,
    #     "num_agents": 4,
    #     # "dataset": ["mnist", "kmnist", "fashionmnist"],
    #     "dataset": "mnist",
    #     "dataset.num_trains_per_class": 64,
    #     "dataset.num_vals_per_class": 50,
    #     "dataset.remap_labels": True,
    #     "dataset.with_replacement": True,
    #     "dataset.num_tasks": 10,
    #     "net": "mlp",
    #     "net.depth": 4,
    #     "net.num_init_tasks": 4,
    #     "net.dropout": 0.5,
    #     "train.num_epochs": 100,
    #     "train.component_update_freq": 100,
    #     "root_save_dir": "results",
    # }
    # main(config)

    config = {
        "algo": ["monolithic", "modular"],
        "seed": 0,
        "num_agents": 4,
        # "dataset": ["mnist", "kmnist", "fashionmnist"],
        "dataset": "mnist",
        "dataset.num_trains_per_class": -1,
        "dataset.num_vals_per_class": -1,
        "dataset.remap_labels": True,
        "dataset.with_replacement": True,
        "dataset.num_tasks": 10,
        "net": "mlp",
        "net.depth": 4,
        "net.num_init_tasks": 4,
        "net.dropout": 0.5,
        "train.num_epochs": 100,
        "train.component_update_freq": 100,
        "root_save_dir": "full_data_results",
    }
    main(config)

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

    # main(config)

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

    # main(config)
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")
