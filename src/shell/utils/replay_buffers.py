'''
File: /replay_buffers.py
Project: utils
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import hashlib


class ReplayBufferBase(TensorDataset):
    def __init__(self, memory_size):
        self.observed = 0
        self.memory_size = memory_size
        self.is_dataset_init = False

    def push(self, *tensors):
        raise NotImplementedError(
            'Implementation must be specific to each buffer type')

    def __len__(self):
        return min(self.observed, self.memory_size)


class ReplayBufferReservoir(ReplayBufferBase):
    def __init__(self, memory_size, task_id, hash=False):
        super().__init__(memory_size)
        self.task_id = task_id
        self.hash = hash
        self.hash_set = set()  # Initialize an empty set to store hashes

    def _compute_hash(self, X):
        # Compute a hash for a tensor. Here, we use a simple approach by hashing the byte representation.
        return hashlib.sha256(X.numpy().tobytes()).hexdigest()

    def get_tensors(self):
        assert self.is_dataset_init, "Replay buffer is not initialized"
        # return tensors but with self.__len__ length
        return (self.tensors[0][:len(self)], self.tensors[1][:len(self)],
                self.tensors[2][:len(self)])

    def push(self, X, Y):
        if not self.is_dataset_init:
            self.tensors = (torch.empty(self.memory_size, *X.shape[1:], device=X.device, dtype=X.dtype),
                            torch.empty(
                                self.memory_size, *Y.shape[1:], device=Y.device, dtype=Y.dtype),
                            torch.full((self.memory_size,), self.task_id, dtype=int))
            if Y.dtype == torch.float32 or self.tensors[1].dtype == torch.float32:
                print(">>>>> WTF???? Y:", Y)
                raise ValueError("WTF????")
            self.is_dataset_init = True

        for j, (x, y) in enumerate(zip(X, Y)):
            if self.hash:  # Only compute and manage hashes if deduplication is enabled
                x_hash = self._compute_hash(x)
                if x_hash in self.hash_set:
                    continue
                self.hash_set.add(x_hash)

            # Add or replace logic
            if self.observed < self.memory_size:
                self.tensors[0][self.observed] = x
                self.tensors[1][self.observed] = y
                # save a picture of x
                self.observed += 1  # Increment here within the bounds check
            else:
                # Ensure replacement happens within bounds
                idx = np.random.randint(0, self.memory_size)
                if self.hash:
                    old_hash = self._compute_hash(self.tensors[0][idx])
                    self.hash_set.remove(old_hash)
                self.tensors[0][idx] = x
                self.tensors[1][idx] = y
