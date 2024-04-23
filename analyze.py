'''
File: /analyze.py
Project: learning-hive
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.utils.record import Record
from shell.utils.metric import Metric
import re
import os


def get_save_dirs(result_dir):
    save_dirs = []
    for job_name in os.listdir(result_dir):
        use_contrastive = "contrastive" in job_name
        for dataset_name in os.listdir(os.path.join(result_dir, job_name)):
            for algo in os.listdir(os.path.join(result_dir, job_name, dataset_name)):
                for seed in os.listdir(os.path.join(result_dir, job_name, dataset_name, algo)):
                    for agent_id in os.listdir(os.path.join(result_dir, job_name, dataset_name, algo, seed)):
                        if agent_id == "hydra_out" or agent_id == "agent_69420":
                            continue
                        save_dir = {'path': os.path.join(
                            result_dir, job_name, dataset_name, algo, seed, agent_id),
                            "dataset": dataset_name,
                            "algo": algo,
                            "use_contrastive": use_contrastive,
                            "seed": seed,
                            "agent_id": agent_id,
                        }
                        save_dirs.append(save_dir)

    return save_dirs


def analyze_save_dirs(save_dirs, pattern=None, num_init_tasks=4, name="result"):
    if pattern is None:
        pattern = r".*"

    record = Record(f"{name}.csv")
    for save_dir in save_dirs:
        # if the pattern doesn't match, continue
        path = save_dir.pop("path")
        if not re.search(pattern, path):
            print('SKIPPING', path)
            continue

        m = Metric(path, num_init_tasks)
        record.write(
            {

                "final_acc": m.compute_final_accuracy(),
                "auc": m.compute_auc(mode='avg'),
            } | save_dir
        )

    record.save()
    return record


def analyze(result_dir):
    save_dirs = get_save_dirs(result_dir)
    record = analyze_save_dirs(save_dirs, name=result_dir)
    return record


if __name__ == "__main__":
    # root_save_dir = "experiment_results"
    # vanilla_dir = "vanilla_jorge_setting_basis_no_sparse"

    # root_save_dir = "budget_experiment_results/jorge_setting_recv_variable_shared_memory_size"
    # vanilla_dir = "mem_size_300_comm_freq_9_num_queries_30"
    # result_dir = os.path.join(root_save_dir, vanilla_dir)

    result_dir = "best_fl_results/fedprox_mu_0.001_comm_freq_5"
    record = analyze(result_dir)
    print("=====FINAL ACC======")
    print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
        "final_acc"].mean() * 100)
    print("=====AUC======")
    print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
        "auc"].mean())
