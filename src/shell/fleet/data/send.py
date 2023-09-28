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
from torch.utils.data import Subset
from shell.datasets.datasets import get_custom_tensordataset

class SendDataAgent(Agent):
    def prepare_communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            for neighbor in self.neighbors:
                self.prepare_data(task_id, neighbor.node_id)
        else:
            self.prepare_feedback()
    
    def communicate(self, task_id, communication_round):
        if communication_round % 2 == 0:
            for neighbor in self.neighbors:
                neighbor.receive(self.node_id, self.outgoing_data[neighbor.node_id], "data")
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
            self.regressors[str(neighbor.node_id)] = nn.Linear(input_size, output_size)
        
        self.device = device
        self.to(device)
        self.regressor_optim = torch.optim.Adam(self.parameters())
    
    def forward(self, X, neighbor_id):
        return self.regressors[str(neighbor_id)](X)



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



    def receive(self, sender_id, data, data_type):
        if data_type == "feedback":
            self.incoming_feedback[sender_id] = data
        elif data_type == "data":
            self.incoming_data[sender_id] = data
        else:
            raise ValueError("Invalid data type")


    def init_recommendation_engine(self):
        self.recommender = RecommenderHeads(input_size=self.net.get_hidden_size(),
                neighbors=self.neighbors)

        # init exploration strategy
        self.exploration = get_exploration(self.sharing_strategy.exploration_strategy)(
          self.neighbors, self.sharing_strategy
        )

        self.regression_loss = nn.MSELoss()
        self.outgoing_data = {}
        self.incoming_feedback = {}
        self.incoming_data = {}
    
    def get_task_candidate_data(self, task_id):
        """
        Get the data in the replay_buffer if the task is already seen
        Otherwise, if the current task is task_id, get the data from training set
        Return a dataset with (X, Y, task_id)
        """
        if task_id == self.agent.T - 1:
            XY = self.dataset.trainset[task_id].tensors
        else:
            XY = self.agent.replay_buffers[task_id].tensors
        
        dataset = torch.utils.data.TensorDataset(*XY)
        # add the task_id as the last tensor. Replay buffer already has this feature
        if task_id == self.agent.T - 1:
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
            indx_tensors = [ds.tensors[indx] for ds in concat_dataset.datasets]
            return torch.cat(indx_tensors, dim=0)  # Modify accordingly if a different dimension is intended
        scores = self.get_pred_scores(concat_tensors(dataset), task_id, neighbor_id)
        selected_data_idx = self.exploration.get_action(scores.detach().cpu(), neighbor_id,
                                                        self.sharing_strategy.bandwidth)

        selected_subset = Subset(dataset, selected_data_idx)

        # save these info 
        self.outgoing_data[neighbor_id] = selected_subset
        return {
            "data": selected_subset,
            "metadata": {
                "class_sequence": self.dataset.class_sequence,
            }
        }


    def get_pred_scores(self, X,
                            task_id,
                            neighbor_id,
                            detach=True):
        X_encode = self.net.encode(X.to(self.net.device), task_id)
        if detach:
            X_encode = X_encode.detach()
        Y_pred = self.recommender(X_encode, neighbor_id).squeeze()
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
    
    def prepare_feedback(self, neighbor_id):
        # process_data should already does this.
        dataset = self.incoming_data[neighbor_id]
        pass



    def process_feedback(self, gt_scores, neighbor_id):
        """
        Update the regressor using the feedback from the neighbors.
        """
        
        # Assuming you have stored outgoing_data and pred_scores
        outgoing_data = self.outgoing_data[neighbor_id]
        assert len(outgoing_data) == len(gt_scores), f"len(outgoing_data)={len(outgoing_data)} != len(gt_scores)={len(gt_scores)}"
        
        data_loader = torch.utils.data.DataLoader(outgoing_data, batch_size=64, shuffle=True)

        optimizer = self.recommender.regressor_optim
        
        start_idx = 0
        for X, _, t in data_loader:
            if isinstance(X, list):
                # contrastive two views
                X = torch.cat([X[0], X[1]], dim=0)
            loss = 0.
            batch_size = len(X)
            Y = gt_scores[start_idx:start_idx + batch_size]  # Extract the corresponding batch from gt_scores
            start_idx += batch_size  # Move the start index for the next batch
        


            if self.agent.use_contrastive:
                Xhaf = X[:len(X)//2]
                Xother = X[len(X)//2:]

            for task_id in torch.unique(t):
                print(t == task_id)
                if self.agent.use_contrastive:
                    # use the original view to train
                    # the recommender
                    Xt_haf = Xhaf[t == task_id]
                    Xt_other = Xother[t == task_id]
                    Xt = Xt_haf
                else:
                    Xt = X[t == task_id]

                Yt = Y[t == task_id]

                Xt = Xt.to(self.recommender.device)
                Yt = Yt.to(self.recommender.device)

                loss += self.compute_recommendation_loss(Xt, Yt,
                                                        task_id,
                                                        neighbor_id)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Clear the stored data and scores as they have been used for training now
        self.outgoing_data[neighbor_id] = None
