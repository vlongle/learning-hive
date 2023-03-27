'''
File: /mlp.py
Project: models
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 i_size,
                 layer_size,
                 depth,
                 num_classes,
                 num_tasks,
                 num_init_tasks,
                 device='cuda',
                 dropout=0.5,
                 init_ordering_mode=None,
                 ):
        super().__init__()
        self.device = device
        self.size = layer_size
        self.depth = depth
        self.num_tasks = num_tasks
        self.num_init_tasks = num_init_tasks
        if isinstance(num_classes, int):
            num_classes = [num_classes] * self.num_tasks

        if isinstance(i_size, int):
            i_size = [i_size] * num_tasks
        self.i_size = i_size

        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.random_linear_projection = nn.Linear(
            self.i_size[0] * self.i_size[0], self.size)
        # freeze the random linear projection
        for param in self.random_linear_projection.parameters():
            param.requires_grad = False

        for i in range(self.depth):
            fc = nn.Linear(self.size, self.size)
            self.components.append(fc)

        self.decoder = nn.ModuleList()
        for t in range(self.num_tasks):
            decoder_t = nn.Linear(
                self.size, num_classes[t])
            self.decoder.append(decoder_t)

        self.to(self.device)

    def load_and_freeze_random_linear_projection(self, state_dict):
        self.random_linear_projection.load_state_dict(state_dict)
        for param in self.random_linear_projection.parameters():
            param.requires_grad = False

    def preprocess(self, X):
        # if X shape is (b, c, h, w) then flatten to (b, c*h*w)
        if len(X.shape) > 2:
            X = X.view(X.shape[0], -1)
        return self.random_linear_projection(X)

    def encode(self, X, task_id):
        X = self.preprocess(X)
        for fc in self.components:
            X = self.dropout(self.relu(fc(X)))
        return X

    def contrastive_embedding(self, X, task_id):
        """
        NOTE: not currently using any projector!
        """
        X = self.encode(X, task_id)
        return X

    def forward(self, X, task_id):
        X = self.encode(X, task_id)
        return self.decoder[task_id](X)
