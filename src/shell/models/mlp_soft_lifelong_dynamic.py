'''
File: /mlp_soft_lifelong_dynamic.py
Project: models
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch
import torch.nn as nn
import numpy as np
import copy
from shell.models.base_net_classes import SoftOrderingNet


class MLPSoftLLDynamic(SoftOrderingNet):
    def __init__(self,
                 i_size,
                 layer_size,
                 depth,
                 num_classes,
                 num_tasks,
                 num_init_tasks=None,
                 max_components=-1,
                 init_ordering_mode='one_module_per_task',
                 device='cuda',
                 freeze_encoder=True,
                 dropout=0.5,
                 ):
        super().__init__(i_size,
                         depth,
                         num_classes,
                         num_tasks,
                         num_init_tasks=num_init_tasks,
                         init_ordering_mode=init_ordering_mode,
                         device=device)
        self.size = layer_size
        self.max_components = max_components if max_components != -1 else np.inf
        self.num_components = self.depth
        self.freeze_encoder = freeze_encoder

        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        for i in range(self.depth):
            fc = nn.Linear(self.size, self.size)
            self.components.append(fc)

        self.decoder = nn.ModuleList()
        for t in range(self.num_tasks):
            decoder_t = nn.Linear(
                self.size, self.num_classes[t])
            self.decoder.append(decoder_t)

        self.to(self.device)

    def add_tmp_module(self, task_id):
        if self.num_components < self.max_components:
            for t in range(self.num_tasks):
                self.structure[t].data = torch.cat((self.structure[t].data, torch.full(
                    (1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
            fc = nn.Linear(self.size, self.size).to(self.device)
            self.components.append(fc)
            self.num_components += 1

    def hide_tmp_module(self):
        self.num_components -= 1

    def recover_hidden_module(self):
        self.num_components += 1

    def remove_tmp_module(self):
        for s in self.structure:
            s.data = s.data[:-1, :]
        self.components = self.components[:-1]
        self.num_components = len(self.components)

    def encode(self, X, task_id):
        if len(X.shape) > 2:
            X = X.view(X.shape[0], -1)
        n = X.shape[0]
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            for j in range(self.num_components):
                fc = self.components[j]
                X_tmp += s[j, k] * self.dropout(self.relu(fc(X)))
            X = X_tmp
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
