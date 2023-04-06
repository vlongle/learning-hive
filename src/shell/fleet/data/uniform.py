'''
File: /uniform.py
Project: data
Created Date: Thursday April 6th 2023
Author: Jason Xie (jchunx@seas.upenn.edu)

Copyright (c) 2023 Jason Xie
'''


from shell.fleet.fleet import Agent
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch
import numpy as np


class UniformDataAgent(Agent):
    
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs,
                 sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.incoming_data = {}

    def prepare_communicate(self, task_id, communication_round):
        self.data = self.get_candidate_data(task_id)

    def communicate(self, task_id, communication_round):
        num_data_points = self.sharing_strategy.num_data_points
        for neighbor in self.neighbors:
            # pick num_data_points data points to send
            selected_indices = np.random.choice(
                len(self.data), num_data_points, replace=False)
            selected_data = Subset(self.data, selected_indices)
            neighbor.receive(self.node_id, selected_data, "data")
            
    def learn_from_recv_data(self, num_epochs, task_id, testloaders, save_freq=1, eval_bool=True):
        
        dataset = ConcatDataset([d for _, d in self.incoming_data.items()])
        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=0, 
                                pin_memory=True)
        self.agent._train(dataloader, num_epochs, task_id,
                    testloaders, save_freq, eval_bool)

    def process_communicate(self, task_id, communication_round):
        testloaders = {task: DataLoader(testset,
                                        batch_size=128,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True,
                                        ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
        task_id_retrain = f"{task_id}_retrain_round_{communication_round}"
        self.learn_from_recv_data(
            self.sharing_strategy.retrain.num_epochs, task_id_retrain, testloaders)

    def get_candidate_data(self, task_id, mode='current'):
        if mode == 'all':
            return ConcatDataset([self.replay_buffer[t] for t in range(task_id+1)])
        elif mode == 'current':
            return self.replay_buffer[task_id]
        else:
            raise ValueError('mode must be either all or current')

    def receive(self, sender_id, data, data_type):
        if data_type == "data":
            # add data to buffer
            self.incoming_data[sender_id] = data
        else:
            raise ValueError("Invalid data type")