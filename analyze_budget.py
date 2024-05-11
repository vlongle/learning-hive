'''
File: /analyze.py
Project: learning-hive
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
'''
File: /plot.py
Project: lifelong-learning-viral
Created Date: Wednesday March 15th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

"""
For cifar100, epochs=500 is stored in
"""


# root_result_dir = "budget_experiment_results/jorge_setting_recv_variable_shared_memory_size"
# root_result_dir = "combined_recv_remove_neighbors_results"
# root_result_dir = "topology_experiment_results/modmod"
# root_result_dir = "topology_experiment_results/topology_experiment_results/jorge_setting_fedavg/comm_freq_5"

# root_result_dir = "budget_experiment_results/jorge_setting_fedavg"
# root_result_dir = "budget_experiment_results/modmod"




import os
import re
from shell.utils.metric import Metric
from  shell.utils.record import Record
def analyze_multiple(root_result_dir, num_init_tasks=4, pattern=r".*"):
    record = Record(f"{root_result_dir}.csv")
    for result_dir in os.listdir(root_result_dir):
        for job_name in os.listdir(os.path.join(root_result_dir, result_dir)):
            use_contrastive = "contrastive" in job_name
            for dataset_name in os.listdir(os.path.join(root_result_dir, result_dir, job_name)):
                for algo in os.listdir(os.path.join(root_result_dir, result_dir, job_name, dataset_name)):
                    for seed in os.listdir(os.path.join(root_result_dir, result_dir, job_name, dataset_name, algo)):
                        for agent_id in os.listdir(os.path.join(root_result_dir, result_dir, job_name, dataset_name, algo, seed)):
                            if agent_id == "hydra_out" or agent_id == "agent_69420":
                                continue
                            save_dir = os.path.join(root_result_dir,
                                                    result_dir, job_name, dataset_name, algo, seed, agent_id)
                            # if the pattern doesn't match, continue
                            if not re.search(pattern, save_dir):
                                continue

                            m = Metric(save_dir, num_init_tasks)
                            # extra_algo = f"{result_dir}_{algo}"
                            extra_algo = f"{algo}_{result_dir}"
                            # print(save_dir, 'algo', extra_algo)
                            record.write(
                                {
                                    "dataset": dataset_name,
                                    "algo": extra_algo,
                                    "use_contrastive": use_contrastive,
                                    "seed": seed,
                                    "agent_id": agent_id,
                                    "avg_acc": m.compute_avg_accuracy(),
                                    "final_acc": m.compute_final_accuracy(),
                                    "auc": m.compute_auc(),
                                }
                            )

                        # print('record', record.df)
                        # exit(0)

    record.save()
    return record


if __name__ == "__main__":
    # root_result_dir = "rerun_fashionmnist_recv_results"
    # root_result_dir = "modular_backward_cifar_heuristic_results_small_mem_32/budget"
    # root_result_dir = "new_topology_experiment_results/modmod"

    # for root_result_dir in os.listdir("budget_and_topology_fedprox_results"):
    #     result = f"budget_and_topology_fedprox_results/{root_result_dir}"
    #     if os.path.isdir(result):
    #         record = analyze_multiple(result)
    #         record.save()

    root_result_dir = "combine_modes_results"
    record = analyze_multiple(root_result_dir)
    print(record.df)
    # get the final accuracy with respect to different algo and dataset
    # and whether it uses contrastive loss
    print("=====FINAL ACC======")
    print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
        "final_acc"].mean() * 100)
    # print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
    #       "final_acc"].sem() * 100)

    print("=====AUC======")
    print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
        "auc"].mean())

    # record.save()
