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
                 init_ordering_mode='random_onehot',
                 device='cuda',
                 dropout=0.5,
                 use_contrastive=None,
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

        self.components = nn.ModuleList()

        self.candidate_modules = nn.ModuleList()
        self.active_candidate_index = None  # Initialize as no active candidate modules
        self.candidate_indices = []  # To hold indices of candidate modules in self.components



        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.random_linear_projection = nn.Linear(
            self.i_size[0] * self.i_size[0], self.size)
        
        self.candidate_indices = []  # To hold indices of candidate modules in self.components

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

        self.to(self.device)

    def load_and_freeze_random_linear_projection(self, state_dict):
        self.random_linear_projection.load_state_dict(state_dict)
        for param in self.random_linear_projection.parameters():
            param.requires_grad = False

    def add_tmp_module(self, task_id):
        if self.num_components < self.max_components:
            for t in range(self.num_tasks):
                self.structure[t].data = torch.cat((self.structure[t].data, torch.full(
                    (1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
            fc = nn.Linear(self.size, self.size).to(self.device)
            self.components.append(fc)
            self.num_components += 1

    def add_tmp_modules(self, task_id, num_modules):
        self.active_candidate_index = None  # Initialize as no active candidate modules
        self.candidate_indices = []  # To hold indices of candidate modules in self.components
        
        for _ in range(num_modules):
            if self.num_components < self.max_components:
                for t in range(self.num_tasks):
                    self.structure[t].data = torch.cat((self.structure[t].data, torch.full(
                        (1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
                fc = nn.Linear(self.size, self.size).to(self.device)
                self.components.append(fc)
                self.candidate_indices.append(self.num_components)
                
                if self.active_candidate_index is None:  # Activate the first candidate by default
                    self.active_candidate_index = self.num_components
                    
                self.num_components += 1


 
    def hide_tmp_module(self):
        self.num_components -= 1

    def hide_tmp_modulev2(self):
        if self.active_candidate_index is not None:
            self.last_active_candidate_index = self.active_candidate_index
        self.active_candidate_index = None  # Deactivating any active candidate module


    def recover_hidden_module(self):
        self.num_components += 1
    
    def recover_hidden_modulev2(self):
        if self.candidate_indices:
            if self.last_active_candidate_index is None:
                self.active_candidate_index = self.candidate_indices[0]
            else:
                # Find the next active candidate index in a round-robin manner
                idx = self.candidate_indices.index(self.last_active_candidate_index)
                self.active_candidate_index = self.candidate_indices[(idx + 1) % len(self.candidate_indices)]
            self.last_active_candidate_index = self.active_candidate_index


    def remove_tmp_module(self):
        for s in self.structure:
            s.data = s.data[:-1, :]
        self.components = self.components[:-1]
        self.num_components = len(self.components)

    def remove_tmp_modulev2(self, excluded_candidate_list):
        for idx in sorted(excluded_candidate_list, reverse=True):  # Sort in reverse to avoid index shifting
            if idx < len(self.candidate_indices):
                # Update components and structure data
                del self.components[self.candidate_indices[idx]]
                for s in self.structure:
                    s.data = torch.cat((s.data[:self.candidate_indices[idx], :], s.data[self.candidate_indices[idx] + 1:, :]), dim=0)
                
                # Update candidate indices
                del self.candidate_indices[idx]
                self.num_components -= 1


    def preprocess(self, X):
        # if X shape is (b, c, h, w) then flatten to (b, c*h*w)
        if len(X.shape) > 2:
            X = X.view(X.shape[0], -1)
        return self.random_linear_projection(X)

    def encode(self, X, task_id):
        X = self.preprocess(X)
        n = X.shape[0]
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            for j in range(self.num_components):
                fc = self.components[j]
                X_tmp += s[j, k] * self.dropout(self.relu(fc(X)))
            X = X_tmp
        return X


    def encodev2(self, X, task_id):
        X = self.preprocess(X)
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            for j in range(self.num_components):
                if j not in self.candidate_indices or j == self.active_candidate_index:  # Allow only active candidate
                    # or the ones that are not candidate modules (not in candidate_indices)
                    fc = self.components[j]
                    X_tmp += s[j, k] * self.dropout(self.relu(fc(X)))
            X = X_tmp
        return X


    def contrastive_embedding(self, X, task_id):
        """
        NOTE: not currently using any projector!
        """
        # X = self.encode(X, task_id)
        X = self.encodev2(X, task_id)
        return X

    def forward(self, X, task_id):
        # X = self.encode(X, task_id)
        X = self.encodev2(X, task_id)
        return self.decoder[task_id](X)
