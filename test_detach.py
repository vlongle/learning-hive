'''
File: /test_detach.py
Project: learning-hive
Created Date: Wednesday April 12th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch


W = torch.zeros((3, 2))
W.requires_grad = True
W.requires_grad = False
b = torch.ones((2, 3))
loss = W @ b

print(loss, loss.requires_grad)
