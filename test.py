'''
File: /test.py
Project: learning-hive
Created Date: Friday March 24th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch.nn as nn
import torch

a = torch.zeros((2, 3))
print(a.requires_grad)  # False
# b but requires grad
b = torch.zeros((2, 3), requires_grad=True)
print(b.requires_grad)  # True

c = a + b
print(c.requires_grad)  # True

# ln = nn.Linear(
#     28, 50)

# # freeze the random linear projection
# for param in ln.parameters():
#     param.requires_grad = False

# ln2 = nn.Linear(
#     28, 50)

# print(ln.weight, ln.weight.requires_grad)
# ln.load_state_dict(ln2.state_dict())
# print(ln.weight, ln.weight.requires_grad)

# ln2.load_state_dict(ln.state_dict())
# # ln = ln2
# print(ln.weight)
# print(ln2.weight)

# for t in range(-1):
# for t in range(-1):
#     print(t)
