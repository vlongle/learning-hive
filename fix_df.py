'''
File: /fix_df.py
Project: learning-hive
Created Date: Wednesday April 5th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import pandas as pd
import os
# path = "vanilla_results_wo_replacement_2/cifar100_modular_numtrain_256/cifar100/modular/seed_0/agent_3"
path = "vanilla_results_wo_replacement_2/cifar100_modular_numtrain_256_contrastive/cifar100/modular/seed_0/agent_1"
path = os.path.join(path, "record.csv")
# path = "vanilla_results_wo_replacement_2/cifar100_modular_numtrain_256/cifar100/modular/seed_0/agent_1/record.csv"

df = pd.read_csv(path)

print(df.head())
print(df['test_acc'].dtype)
# convert: test_acc, test_loss to numeric
# df['test_acc'] = pd.to_numeric(df['test_acc'], errors='coerce')
# df['test_loss'] = pd.to_numeric(df['test_loss'], errors='coerce')
# # overwrite the file
# df.to_csv(path, index=False)
