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


# result_dir = "results"
# result_dir = "vanilla_results"
# result_dir = "vanilla_2modules_results"
import os
from shell.utils.record import Record
from shell.utils.metric import Metric
import re
result_dir = "vanilla_results"
record = Record("experiment_results.csv")

# pattern = r"/fashion.*"
# pattern = r".*64.*"
# pattern = r".*64_contrastive"
pattern = r".*256.*"

num_init_tasks = 4

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
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "final_acc"].sem())
print("=====AVG ACC======")
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "avg_acc"].mean())

print("=====BACKWARD======")
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "backward"].mean())

print("=====FORWARD======")
print(record.df.groupby(["algo", "dataset", "use_contrastive"])[
      "forward"].mean())

record.save()
