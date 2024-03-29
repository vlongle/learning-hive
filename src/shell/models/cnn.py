'''
File: /cnn.py
Project: models
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self,
                 i_size,
                 channels,
                 depth,
                 num_classes,
                 num_tasks,
                 num_init_tasks,
                 conv_kernel=3,
                 maxpool_kernel=2,
                 padding=0,
                 device='cuda',
                 dropout=0.5,
                 init_ordering_mode=None,
                 use_contrastive=False,
                 no_sparse_basis=True,  # doesn't matter
                 ):
        super().__init__()
        self.device = device
        self.depth = depth
        self.channels = channels
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_init_tasks = num_init_tasks

        if isinstance(i_size, int):
            i_size = [i_size] * num_tasks
        self.i_size = i_size
        if isinstance(self.num_classes, int):
            self.num_classes = [self.num_classes] * num_tasks

        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(maxpool_kernel)
        self.dropout = nn.Dropout(dropout)

        out_h = i_size[0]
        for i in range(self.depth):
            conv = nn.Conv2d(self.channels, self.channels,
                             conv_kernel, padding=padding)
            self.components.append(conv)
            out_h = out_h + 2 * padding - conv_kernel + 1
            out_h = (out_h - maxpool_kernel) // maxpool_kernel + 1

        self.decoder = nn.ModuleList()
        for t in range(self.num_tasks):
            decoder_t = nn.Linear(
                out_h * out_h * self.channels, self.num_classes[t])
            self.decoder.append(decoder_t)

        self.out_h = out_h

        mean = (0.5079, 0.4872, 0.4415)
        std = (0.2676, 0.2567, 0.2765)
        # normalize
        self.transform = transforms.Normalize(mean, std)

        self.use_contrastive = use_contrastive
        if self.use_contrastive:
            hidden_dim = 64
            dim_in = out_h * out_h * self.channels
            self.projector = nn.ModuleList([nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, hidden_dim),
            ) for t in range(self.num_tasks)])
        self.to(self.device)

    def encode(self, X, task_id):
        X = self.transform(X)
        c = X.shape[1]
        X = F.pad(X, (0, 0, 0, 0, 0, self.channels-c))
        for conv in self.components:
            X = self.dropout(self.relu(self.maxpool(conv(X))))

        X = X.reshape(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return X

    def forward(self, X, task_id):
        X = self.encode(X, task_id)
        return self.decoder[task_id](X)

    def contrastive_embedding(self, X, task_id):
        X = self.encode(X, task_id)
        X = self.projector[task_id](X)  # (N, 128)
        X = F.normalize(X, dim=1)
        return X
