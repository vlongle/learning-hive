'''
File: /analyze_no_components.py
Project: learning-hive
Created Date: Friday April 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import pandas as pd
import re
from shell.utils.metric import Metric
from shell.utils.record import Record
import os
# result_dir = "vanilla_results"
# result_dir = "cifar_contrastive_no_dropout_results"
# result_dir = "cifar_no_updates_contrastive_results"
# result_dir = "cifar_contrastive_no_dropout_results"
# result_dir = "cifar_epochs_500_mild_dropout_memory_64_data_300_results"
result_dir = "vanilla_remove_datasets_hack_results"
record = Record(f"{result_dir}.csv")

pattern = r".*"


record = Record(f"{result_dir}_no_components.csv")

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

                    # only makes sense for modular algo
                    if algo != "modular":
                        continue
                    print(save_dir)
                    df = pd.read_csv(os.path.join(
                        save_dir, "add_modules_record.csv"))
                    # get the num_components at last location
                    num_components = df["num_components"].iloc[-1]
                    record.write(
                        {
                            "dataset": dataset_name,
                            "algo": algo,
                            "use_contrastive": use_contrastive,
                            "seed": seed,
                            "agent_id": agent_id,
                            "no_components": num_components,
                        }
                    )

# average out seed and agent_id
print(record.df.groupby(
    ["dataset", "algo", "use_contrastive"])["no_components"].mean().reset_index())
record.save()
