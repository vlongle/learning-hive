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

import os
import re
from shell.utils.metric import Metric
from  shell.utils.record import Record
# result_dir = "/mnt/kostas-graid/datasets/vlongle/experiment_results/tryout_tryout_epochs_100_max_modules_14_leep_jorge_setting_lowest_task_id_wins_modmod_test_sync_base_True_opt_with_random_False_frozen_False_transfer_decoder_True_transfer_structure_True_no_sparse_basis_True"
result_dir = "/mnt/kostas-graid/datasets/vlongle/experiment_results/jorge_setting_lowest_task_id_wins_modmod_test_sync_base_True_opt_with_random_False_frozen_False_transfer_decoder_True_transfer_structure_True_no_sparse_basis_True"

record = Record(f"{result_dir}.csv")

pattern = r".*"


num_init_tasks = 4  # vanilla_results


num_epochs_ = 100
num_init_epochs_ = 300


start_epoch = 21

for job_name in os.listdir(result_dir):
    use_contrastive = "contrastive" in job_name
    for dataset_name in os.listdir(os.path.join(result_dir, job_name)):
        for algo in os.listdir(os.path.join(result_dir, job_name, dataset_name)):
            for seed in os.listdir(os.path.join(result_dir, job_name, dataset_name, algo)):
                for agent_id in os.listdir(os.path.join(result_dir, job_name, dataset_name, algo, seed)):
                    if agent_id == "hydra_out" or agent_id == "agent_69420":
                        continue
                    save_dir = os.path.join(
                        result_dir, job_name, dataset_name, algo, seed, agent_id)
                    # if the pattern doesn't match, continue
                    print('pattern', pattern)
                    if not re.search(pattern, save_dir):
                        continue
                    print(save_dir)

                    num_epochs = num_init_epochs = None
                    if dataset_name == "cifar100":
                        num_epochs = num_epochs_
                        num_init_epochs = num_init_epochs_
                    m = Metric(save_dir, num_init_tasks, num_epochs=num_epochs,
                               num_init_epochs=num_init_epochs)
                    record.write(
                        {
                            "dataset": dataset_name,
                            "algo": algo,
                            "use_contrastive": use_contrastive,
                            "seed": seed,
                            "agent_id": agent_id,
                            "avg_acc": m.compute_avg_accuracy(),
                            "final_acc": m.compute_final_accuracy(),
                            "forward": m.compute_forward_transfer(start_epoch=start_epoch),
                            "backward": m.compute_backward_transfer(),
                            "catastrophic": m.compute_catastrophic_forgetting(),
                        }
                    )
print(record.df)
# get the final accuracy with respect to different algo and dataset
# and whether it uses contrastive loss
print("=====FINAL ACC======")
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "final_acc"].mean() * 100)
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "final_acc"].sem() * 100)
print("=====AVG ACC======")
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "avg_acc"].mean() * 100)

# # print("=====BACKWARD======")
# # print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
# #       "backward"].mean())

# print("=====FORWARD======")
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "forward"].mean() * 100)


# print("=====CATASTROPHIC======")
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "catastrophic"].mean())


record.save()
