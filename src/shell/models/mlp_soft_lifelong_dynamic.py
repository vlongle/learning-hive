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
import torch.nn.functional as F


class MLPSoftLLDynamic(SoftOrderingNet):
    def __init__(self,
                 i_size,
                 layer_size,
                 depth,
                 num_classes,
                 num_tasks,
                 num_init_tasks=None,
                 max_components=-1,
                 init_ordering_mode='random_onehot',
                 device='cuda',
                 dropout=0.5,
                 use_contrastive=None,
                 normalize=False,
                 use_projector=False,
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

        self.normalize = normalize

        self.components = nn.ModuleList()

        self.active_candidate_index = None  # Initialize as no active candidate modules
        self.candidate_indices = []  # To hold indices of candidate modules in self.components
        self.last_active_candidate_index = None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.random_linear_projection = nn.Linear(
            self.i_size[0] * self.i_size[0], self.size)

        # freeze the random linear projection (preprocessing)
        for param in self.random_linear_projection.parameters():
            param.requires_grad = False

        for i in range(self.depth):
            fc = nn.Linear(self.size, self.size)
            self.components.append(fc)

        self.decoder = nn.ModuleList()
        for t in range(self.num_tasks):
            decoder_t = nn.Linear(
                self.size, self.num_classes[t])
            self.decoder.append(decoder_t)

        self.use_contrastive = use_contrastive
        self.use_projector = use_projector
        if self.use_contrastive and self.use_projector:
            self.projector = nn.ModuleList([nn.Linear(self.size, self.size // 2)
                                           for t in range(self.num_tasks)])

        self.to(self.device)

    def get_hidden_size(self):
        return self.size

    def load_and_freeze_random_linear_projection(self, state_dict):
        self.random_linear_projection.load_state_dict(state_dict)
        for param in self.random_linear_projection.parameters():
            param.requires_grad = False

    def add_tmp_modules(self, task_id, num_modules):
        if num_modules == 0:
            return
        self.active_candidate_index = None  # Initialize as no active candidate modules
        self.candidate_indices = []  # To hold indices of candidate modules in self.components

        # print('BEFORE ADDING TMP_MODULES', self.structure)
        for _ in range(num_modules):
            if self.num_components < self.max_components:
                for t in range(self.num_tasks):
                    if self.structure[t].shape[0] < self.num_components + 1:
                        self.structure[t].data = torch.cat((self.structure[t].data, torch.full(
                            (1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
                fc = nn.Linear(self.size, self.size).to(self.device)
                self.components.append(fc)
                self.candidate_indices.append(self.num_components)

                if self.active_candidate_index is None:  # Activate the first candidate by default
                    self.active_candidate_index = self.num_components

                self.num_components += 1

        # verify that structure[t] is of shape (num_components, depth)
        for t in range(self.num_tasks):
            assert self.structure[t].shape[0] == self.num_components
        # print('ADDED TMP MODULES', self.structure[task_id])

    def receive_modules(self, task_id, module_list):
        # Number of temporary modules added in the last step
        num_tmp_modules = len(self.candidate_indices)

        assert len(
            module_list) <= num_tmp_modules, 'Number of modules received must be less than or equal to the number of temporary modules'
        # Loop over the temporary modules, (implicitly excluding the last one due to
        # how base_learning_classes.py is written)
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

    def preprocess(self, X):
        # if X shape is (b, c, h, w) then flatten to (b, c*h*w)
        if len(X.shape) > 2:
            X = X.view(X.shape[0], -1)
        return self.random_linear_projection(X)

    def encode(self, X, task_id):
        X = self.preprocess(X)
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            for j in range(self.num_components):
                if j not in self.candidate_indices or j == self.active_candidate_index:  # Allow only active candidate
                    # or the ones that are not candidate modules (not in candidate_indices)
                    fc = self.components[j]
                    # last ditch attempt
                    out = self.relu(fc(X))
                    # only dropout on non-basis modules
                    if j >= self.num_init_tasks:
                        out = self.dropout(out)
                    X_tmp += s[j, k] * out
            X = X_tmp
        return X

    def contrastive_embedding(self, X, task_id):
        """
        NOTE: not currently using any projector!
        """
        # X = self.encode(X, task_id)
        # # X = self.projector(X)
        # if self.normalize:
        #     X = F.normalize(X, dim=1)
        # return X

        feat = self.encode(X, task_id)
        if self.use_projector:
            feat = self.projector[task_id](feat)
        if self.normalize:
            feat = F.normalize(feat, dim=1)
        return feat

    def forward(self, X, task_id):
        X = self.encode(X, task_id)
        return self.decoder[task_id](X)
