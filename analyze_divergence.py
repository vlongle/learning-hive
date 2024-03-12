'''
File: /analyze_divergence.py
Project: learning-hive
Created Date: Friday April 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import re
from shell.utils.metric import DivergenceMetric
from shell.utils.record import Record
import os
import pandas as pd
result_dir = "experiment_results/jorge_setting_fedavg"

pattern = r".*"

concat_df = pd.DataFrame()
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
                    if not re.search(pattern, save_dir):
                        continue
                    print(save_dir)
                    df = DivergenceMetric(save_dir).df
                    # only keep task_id, communication_round, time, and avg_params
                    # columns
                    df = df[["task_id", "communication_round",
                             "epoch", "avg_params"]]
                    # add seed and agent_id columns
                    df["seed"] = seed
                    df["agent_id"] = agent_id
                    df['algo'] = algo
                    df['dataset'] = dataset_name
                    df['use_contrastive'] = use_contrastive
                    concat_df = pd.concat([concat_df, df])


# reduce concat_df averaging over seed, and agent_id
# to get avg_params (mean) and avg_params_stderr (standard error) and
# avg_params_std (standard deviation)

concat_df = concat_df.groupby(
    ["task_id", "communication_round", "epoch", "algo", "dataset", "use_contrastive"]).agg(
        avg_params=("avg_params", "mean"),
        avg_params_stderr=("avg_params", "sem"),
        avg_params_std=("avg_params", "std")
).reset_index()


# save to csv
concat_df.to_csv(f"{result_dir}_divergence.csv", index=False)
