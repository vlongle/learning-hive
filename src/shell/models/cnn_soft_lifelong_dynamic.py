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
                 init_ordering_mode='random_onehot',
                 device='cuda',
                 dropout=0.5,
                 use_contrastive=False,
                 no_sparse_basis=False,
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

        self.active_candidate_index = None  # Initialize as no active candidate modules
        self.candidate_indices = []  # To hold indices of candidate modules in self.components
        self.last_active_candidate_index = None

        self.no_sparse_basis = no_sparse_basis

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

        # mean = (0.5079, 0.4872, 0.4415)
        # std = (0.2676, 0.2567, 0.2765)
        # normalize
        # self.transform = transforms.Normalize(mean, std)

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

    def add_tmp_modules(self, task_id, num_modules):
        if num_modules == 0:
            return
        self.active_candidate_index = None  # Initialize as no active candidate modules
        self.candidate_indices = []  # To hold indices of candidate modules in self.components

        print('BEFORE ADDING TMP_MODULES', self.structure[task_id].shape, 'no_comp', self.num_components, 'len(comp)', len(self.components))
        for _ in range(num_modules):
            if self.num_components < self.max_components:
                for t in range(self.num_tasks):
                    self.structure[t].data = torch.cat((self.structure[t].data, torch.full(
                        (1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
                conv = nn.Conv2d(self.channels, self.channels,
                                 self.conv_kernel, padding=self.padding).to(self.device)
                self.components.append(conv)
                self.candidate_indices.append(self.num_components)

                if self.active_candidate_index is None:  # Activate the first candidate by default
                    self.active_candidate_index = self.num_components

                self.num_components += 1

        # verify that structure[t] is of shape (num_components, depth)
        for t in range(self.num_tasks):
            if self.structure[t].shape[0] != self.num_components:
               print(f"!!ERR: structure[t].shape = {self.structure[t].shape} != {self.num_components}")
            assert self.structure[t].shape[
                0] == self.num_components, f"structure[t].shape = {self.structure[t].shape} != {self.num_components}"

    def receive_modules(self, task_id, module_list):
        # Number of temporary modules added in the last step
        num_tmp_modules = len(self.candidate_indices)

        assert len(
            module_list) <= num_tmp_modules, 'Number of modules received must be less than or equal to the number of temporary modules'
        # Loop over the temporary modules, excluding the last one
        for i in range(len(module_list)):
            tmp_module_idx = self.candidate_indices[i]
            # Replacing the state_dict of the temporary module with the corresponding one in the module_list
            self.components[tmp_module_idx].load_state_dict(
                module_list[i].state_dict())

    def hide_tmp_modulev2(self):
        if self.active_candidate_index is not None:
            self.last_active_candidate_index = self.active_candidate_index
        self.active_candidate_index = None  # Deactivating any active candidate module

    def get_next_active_candidate_index(self):
        # round-robin selection
        if not self.candidate_indices or not self.last_active_candidate_index:
            return None
        idx = self.candidate_indices.index(self.last_active_candidate_index)
        return self.candidate_indices[(idx + 1) % len(self.candidate_indices)]

    def recover_hidden_modulev2(self):
        self.active_candidate_index = self.last_active_candidate_index

    def select_active_module(self, index=None):
        if index is not None:
            self.active_candidate_index = index
        else:
            self.active_candidate_index = self.get_next_active_candidate_index()

    def remove_tmp_modulev2(self, excluded_candidate_list):
        # Create a new ModuleList and add components that are not in the excluded list
        new_components = nn.ModuleList()
        for idx, component in enumerate(self.components):
            if idx not in self.candidate_indices or idx not in excluded_candidate_list:
                new_components.append(component)

        # Create a new ParameterList and add structure elements that are not in the excluded list
        new_structure = nn.ParameterList()
        for t in range(self.num_tasks):
            rows_to_keep = [idx for idx in range(
                self.num_components) if idx not in excluded_candidate_list]
            # Keeping rows not in the excluded_candidate_list
            new_structure.append(self.structure[t].data[rows_to_keep, :])
        # for idx, structure in enumerate(self.structure):
        #     if idx not in self.candidate_indices or idx not in excluded_candidate_list:
        #         new_structure.append(structure)

        self.num_components -= len(excluded_candidate_list)
        self.components = self.components[:self.num_components]
        for s in self.structure:
            s.data = s.data[:self.num_components, :]

        # Copy the state_dict of new components and structure to the original ones
        self.components.load_state_dict(new_components.state_dict())
        self.structure.load_state_dict(new_structure.state_dict())

        # Update candidate_indices and num_components

        # Reset the round-robin variables
        self.active_candidate_index = None
        self.last_active_candidate_index = None
        self.candidate_indices = []

    def get_hidden_size(self):
        return self.size

    def encode(self, X, task_id):
        # X = self.transform(X)
        c = X.shape[1]
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        X = F.pad(X, (0, 0, 0, 0, 0, self.channels-c, 0, 0))
        for k in range(self.depth):
            X_tmp = 0.
            for j in range(self.num_components):
                if j not in self.candidate_indices or j == self.active_candidate_index:  # Allow only active candidate
                    conv = self.components[j]
                    out = self.relu(self.maxpool(conv(X)))
                    if j >= self.num_init_tasks or not self.no_sparse_basis:
                        out = self.dropout(out)
                    X_tmp += s[j, k] * out
            X = X_tmp
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
