'''
File: /sender.py
Project: fleet
Created Date: Tuesday March 28th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.fleet.fleet import Agent
import torch.nn as nn
import torch
from shell.fleet.data.send_exploration import *
from shell.fleet.data.data_utilize import *
from torch.utils.data.dataset import ConcatDataset
from shell.datasets.datasets import get_custom_tensordataset

class SendDataAgent(Agent):
    def prepare_communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            self.prepare_data()
        else:
            self.prepare_feedback()
    
    def communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.data, "data")
        else:
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.feedback, "feedback")

    def process_communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            self.process_data()
        else:
            self.process_feedback()


class RecommenderHeads(nn.Module):
    def __init__(self, input_size, neighbors, output_size=1,
    device="cuda"):
        super().__init__()
        self.regressors = nn.ModuleDict()
        for neighbor in neighbors:
            self.regressors[neighbor.node_id] = nn.Linear(input_size, output_size)
        self.to(device)
        self.regressor_optim = torch.optim.Adam(self.parameters())



class SendDataAgent(SendDataAgent):
    """
    Even round: send data.
    Odd round: send feedback.
    """

    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
        # remember which data is already received by the neighbors
        # already_taken[neighbor][task] = set of idx of data already taken
        # self.already_taken = {
        #     neighbor: {task: set() for task in range(self.net.num_tasks)} for neighbor in self.neighbors}



    def init_recommendation_engine(self):
        # init the feedback prediction (regressor)
        # self.regressors = {}
        # self.eps = {}
        # for neighbor in self.neighbors:
        #     regressor = nn.Linear(
        #         self.net.get_hidden_size(), 1)
        #     self.regressors[neighbor.node_id] = regressor
        #     self.eps[neighbor.node_id] = self.sharing_strategy.init_eps
        self.recommender = RecommenderHeads(input_size=self.net.get_hidden_size(),
                neighbors=self.neighbors)

        # init exploration strategy
        self.exploration = get_exploration(self.sharing_strategy.exploration_strategy)(
            num_slates=len(self.neighbors)
        )
    
    def get_task_candidate_data(self, task_id):
        """
        Get the data in the replay_buffer if the task is already seen
        Otherwise, if the current task is task_id, get the data from training set
        Return a dataset with (X, Y, task_id)
        """
        # if self.agent.T == task_id-1:
        if self.agent.T == task_id:
            XY = self.dataset.trainset[task_id].tensors
        else:
            print('getting from replay', task_id)
            XY = self.agent.replay_buffers[task_id].tensors
        
        dataset = torch.utils.data.TensorDataset(*XY)
        # add the task_id as the last tensor
        dataset.tensors = dataset.tensors + (torch.full((len(dataset),), task_id, dtype=int),)
        return get_custom_tensordataset(dataset.tensors,
                                        name=self.agent.dataset_name,
                                        use_contrastive=self.agent.use_contrastive,)


    def get_candidate_data(self, task_id, neighbor_id, mode="all"):
        if mode == "all":
            tasks = range(task_id + 1)
        elif mode == "current":
            tasks = [task_id]
        else:
            raise ValueError(f"Invalid mode {mode}")
        
        data = []
        for task_id in tasks:
            print('neighbor_id', neighbor_id, 'task_id', task_id)
            data.append(self.get_task_candidate_data(task_id))

        data = ConcatDataset(data)
        return self.filter_redudant_data(data)
        

    def filter_redudant_data(self, data):
        # TODO: remove data that is already taken by the neighbor
        # # get the data that is not already taken by the neighbor
        # # return a list of indices
        # data = self.replay_buffer[task_id]
        # # get the indices of the data
        # indices = list(range(len(data)))
        # # get the indices that are not already taken
        # candidate_indices = list(
        #     set(indices) - self.already_taken[neighbor_id][task_id])
        # return data[candidate_indices], candidate_indices
        return data


    def prepare_data(self, task_id, neighbor_id):
        """
        Get data from all task <= task_id from 
        self.get_candidate_data. Compute the
        predicted preference scores from self.regressors[neighbor_id].
        Then run self.select() function (which internally uses eps-greedy)
        to select self.sharing_strategy.bandwidth data to send to the neighbor.
        and remember the candidate_idx being selected.
        Data is in the form
        {
            "task_id": task_id,
            "class": class,
            "data": data,
        }
        """
        dataset = self.get_candidate_data(task_id, neighbor_id)
        def concat_tensors(concat_dataset, indx=0):
            """
            Utility function to concatenate the `indx` tensors 
            from all datasets in a ConcatDataset
            """
            first_tensors = [ds.tensors[indx] for ds in concat_dataset.datasets]
            return torch.cat(first_tensors, dim=0)  # Modify accordingly if a different dimension is intended

        scores = self.get_pred_scores(concat_tensors(dataset), task_id, neighbor_id)
        selected_data = self.exploration.get_action(dataset, scores)
        return selected_data

    def get_pred_scores(self, X,
                            task_id,
                            neighbor_id,
                            detach=True):
        X_encode = self.net.encode(X.to(self.net.device), task_id)
        if detach:
            X_encode = X_encode.detach()
        Y_pred = self.regressors[neighbor_id](X_encode)
        return Y_pred

    def compute_recommendation_loss(self,  X, Y,
                                    task_id,
                                    neighbor_id,
                                    detach=True):
        """
        Use the contrastive embedding for faster learning
        but apply a stop gradient to only train the regressor layer
        and stop backpropagating to the embedding layer.
        See `compute_cross_entropy_loss` in `base_learning_classes.py`
        """
        Y_pred = self.get_pred_scores(X, task_id, neighbor_id, detach)
        return self.regression_loss(Y_pred, Y)

    def process_data(self):
        """
        Train the model. Evaluate the data usefulness. 
        Then return the feedback
        """
        pass

    def process_feedback(self):
        """
        Update the regressor using the feedback from the neighbors.
        """
        pass

