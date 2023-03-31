'''
File: /test_custom_tensors.py
Project: learning-hive
Created Date: Thursday March 30th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch
# import tensordataset
from torch.utils.data import TensorDataset


X = torch.zeros(64, 3, 32, 32)
y = torch.zeros(64)
z = torch.zeros(64)

dataset = TensorDataset(X, y, z)
dataset2 = TensorDataset(X, y)

# print(dataset[0])

# print(dataset2[0])

# create a get_tensordataset takes in a list of tensors and returns a TensorDataset


def get_tensordataset(tensors):
    return TensorDataset(*tensors)


# print(get_tensordataset(dataset.tensors))

print(get_tensordataset([X, y, z])[0])
print(get_tensordataset(dataset.tensors)[0])
