'''
File: /cnn_soft_lifelong_dynamic.py
Project: models
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from shell.models.base_net_classes import SoftOrderingNet


class CNNSoftLLDynamic(SoftOrderingNet):
    def __init__(self,
                 i_size,
                 channels,
                 depth,
                 num_classes,
                 num_tasks,
                 conv_kernel=3,
                 maxpool_kernel=2,
                 padding=0,
                 num_init_tasks=None,
                 max_components=-1,
                 init_ordering_mode='one_module_per_task',
                 device='cuda',
                 dropout=0.5,
                 ):
        super().__init__(i_size,
                         depth,
                         num_classes,
                         num_tasks,
                         num_init_tasks=num_init_tasks,
                         init_ordering_mode=init_ordering_mode,
                         device=device)
        self.channels = channels
        self.conv_kernel = conv_kernel
        self.padding = padding
        self.max_components = max_components if max_components != -1 else np.inf
        self.num_components = self.depth

        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(maxpool_kernel)
        self.dropout = nn.Dropout(dropout)

        out_h = self.i_size[0]
        for i in range(self.depth):
            conv = nn.Conv2d(channels, channels, conv_kernel, padding=padding)
            self.components.append(conv)
            out_h = out_h + 2 * padding - conv_kernel + 1
            out_h = (out_h - maxpool_kernel) // maxpool_kernel + 1
        self.decoder = nn.ModuleList()
        for t in range(self.num_tasks):
            decoder_t = nn.Linear(
                out_h * out_h * channels, self.num_classes[t])
            self.decoder.append(decoder_t)

        mean = (0.5079, 0.4872, 0.4415)
        std = (0.2676, 0.2567, 0.2765)
        # normalize
        self.transform = transforms.Normalize(mean, std)

        hidden_dim = 128
        # self.projector = nn.Linear(out_h * out_h * self.channels,
        #                            hidden_dim)
        dim_in = out_h * out_h * self.channels
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, hidden_dim)
        )
        self.to(self.device)

    def add_tmp_module(self, task_id):
        if self.num_components < self.max_components:
            for t in range(self.num_tasks):
                self.structure[t].data = torch.cat((self.structure[t].data, torch.full(
                    (1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
            conv = nn.Conv2d(self.channels, self.channels,
                             self.conv_kernel, padding=self.padding).to(self.device)
            self.components.append(conv)
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
        X = self.transform(X)
        c = X.shape[1]
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        X = F.pad(X, (0, 0, 0, 0, 0, self.channels-c, 0, 0))
        for k in range(self.depth):
            X_tmp = 0.
            for j in range(self.num_components):
                conv = self.components[j]
                out = self.dropout(self.relu(self.maxpool(conv(X))))
                X_tmp += s[j, k] * out
                print('task_id', task_id, 'j', j,
                      'k', k, 'contr', torch.sum(s[j, k] * out))
            X = X_tmp
        X = X.reshape(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return X

    def forward(self, X, task_id):
        X = self.encode(X, task_id)
        return self.decoder[task_id](X)

    def contrastive_embedding(self, X, task_id):
        """
        NOTE: not currently using any projector!
        """
        X = self.encode(X, task_id)
        X = self.projector(X)  # (N, 128)
        X = F.normalize(X, dim=1)
        return X
