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

# result_dir = "finding_hyper_for_mod_contrastive_large_deeper_projector_results"
# result_dir = "cifar_lasttry_im_done_projector_no_freeze_results"
# result_dir = "cifar_lasttry_im_done_results"
# result_dir = "vanilla_results_wo_replacement_2"
# result_dir = "vanilla_results_wo_replacement"
# result_dir = "vanilla_results_wo_replacement_2"
# result_dir = "finding_hyper_for_mod_contrastive_large_results"
# result_dir = "finding_hyper_for_mod_contrastive_large_lower_temp_results"
# result_dir = "vanilla_init_big_mod_nocontrast_results"
# result_dir = "vanilla_cifar_old_results"
# result_dir = "vanilla_results"
import os
from shell.utils.record import Record
from shell.utils.metric import Metric
import re
result_dir = "grad_results"
# result_dir = "finding_hyper_for_mod_contrastive2"
record = Record("experiment_results.csv")

# pattern = r"/fashion.*"
# pattern = r".*64.*"
pattern = r".*"
# pattern = r".*64_contrastive"
# pattern = r".*256.*"
# pattern = r".*modular_numtrain_256_contrastive.*"

# TODO: BUG: be careful with num_init_tasks!!!!!
# TODO:
# TODO: this might not be true for config that involves joint training
# e.g., grad and mod^2
# num_init_tasks = 4
num_init_tasks = 0

for job_name in os.listdir(result_dir):
    use_contrastive = "contrastive" in job_name
    for dataset_name in os.listdir(os.path.join(result_dir, job_name)):
        for algo in os.listdir(os.path.join(result_dir, job_name, dataset_name)):
            for seed in os.listdir(os.path.join(result_dir, job_name, dataset_name, algo)):
                for agent_id in os.listdir(os.path.join(result_dir, job_name, dataset_name, algo, seed)):
                    if agent_id == "hydra_out":
                        continue
                    save_dir = os.path.join(
                        result_dir, job_name, dataset_name, algo, seed, agent_id)
                    # if the pattern doesn't match, continue
                    if not re.search(pattern, save_dir):
                        continue
                    print(save_dir)

                    m = Metric(save_dir, num_init_tasks)
                    record.write(
                        {
                            "dataset": dataset_name,
                            "algo": algo,
                            "use_contrastive": use_contrastive,
                            "seed": seed,
                            "agent_id": agent_id,
                            "avg_acc": m.compute_avg_accuracy(),
                            "final_acc": m.compute_final_accuracy(),
                            "forward": m.compute_forward_transfer(),
                            "backward": m.compute_backward_transfer(
                            ),
                        }
                    )
print(record.df)
# get the final accuracy with respect to different algo and dataset
# and whether it uses contrastive loss
print("=====FINAL ACC======")
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "final_acc"].mean())
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "final_acc"].sem())
# print("=====AVG ACC======")
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "avg_acc"].mean())

# print("=====BACKWARD======")
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "backward"].mean())

# print("=====FORWARD======")
# print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
#       "forward"].mean())

record.save()
