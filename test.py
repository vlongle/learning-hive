'''
File: /test.py
Project: learning-hive
Created Date: Friday March 24th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch.nn as nn


ln = nn.Linear(
    28, 50)

# freeze the random linear projection
for param in ln.parameters():
    param.requires_grad = False

ln2 = nn.Linear(
    28, 50)

ln2.load_state_dict(ln.state_dict())
# ln = ln2
print(ln.weight)
print(ln2.weight)
