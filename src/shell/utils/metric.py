'''
File: /metric.py
Project: utils
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import pandas as pd
import os
import logging


class Metric:
    """
    4 basic lifelong learning metrics:
    - Avg accuracy: avg final accuracy of past tasks at the end of each training task.
    - Final accuracy: final accuracy over all tasks at the end of lifetime.
    - Forward transfer (zeroshot): accuracy on a new task at epoch = 0 (no training).
    - Backward transfer: the difference accuracy of past task at epoch = 0 and after training
    on the current task.

    dataframe has the following columns:
    - train_task: task id of the current training task
    - test_task: task id of the test task. ("avg" means average of all past tasks including
    the current training task)
    - epoch: training epoch of the current training task
    - test_acc: test accuracy of the test task (numeric)
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, 'record.csv')
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
        else:
            logging.critical(f"File {self.file_path} does not exist")

        self.max_epoch = self.df['epoch'].max()
        # 'test_task' column is string, we need to convert train_task to string as well
        self.df['train_task'] = self.df['train_task'].astype(str)

    def compute_avg_accuracy(self, reduce='mean'):
        """
        Compute the average accuracy of past tasks at the end of each training task.
        """
        avg_acc = self.df[(self.df['test_task'] == 'avg') & (
            self.df['epoch'] == self.max_epoch)]
        if reduce == 'mean':
            avg_acc = avg_acc['test_acc'].mean()
        return avg_acc

    def compute_final_accuracy(self, reduce='mean'):
        """
        Compute the final accuracy over all tasks at the end of lifetime.
        """
        max_task = self.df['train_task'].max()
        final_acc = self.df[(self.df['train_task'] == max_task) & (
            self.df['epoch'] == self.max_epoch)]
        if reduce == 'mean':
            final_acc = final_acc[final_acc['test_task']
                                  == 'avg']['test_acc'].item()
        return final_acc

    def compute_forward_transfer(self, start_epoch=0, reduce='mean'):
        """
        Compute the forward transfer (zeroshot) accuracy on a new task at epoch = start_epoch
        start_epoch = 0 means no training.
        """
        forward_transfer = self.df[(self.df['epoch'] == start_epoch) & (
            self.df['test_task'] == self.df['train_task'])]
        if reduce == 'mean':
            forward_transfer = forward_transfer['test_acc'].mean()
        return forward_transfer

    def compute_backward_transfer(self, num_init_tasks=None, reduce='mean'):
        """
        Compute the backward transfer: the difference accuracy of past task at epoch = 0 and after training
        on the current task.
        """
        before_train = self.df[(self.df['epoch'] == 0)]
        after_train = self.df[(self.df['epoch'] == self.max_epoch)]
        # take the difference between after_train and before_train
        # matching on train_task and test_task
        backward_transfer = after_train.merge(before_train, on=[
            'train_task', 'test_task'], suffixes=('_after', '_before'))
        backward_transfer['backward_transfer'] = (backward_transfer['test_acc_after'] -
                                                  backward_transfer['test_acc_before'])
        # backward transfer is with respect to past tasks, so remove column "test_task" == "avg"
        # or "test_task" == "train_task"
        backward_transfer = backward_transfer[(backward_transfer['test_task'] != 'avg') & (
            backward_transfer['test_task'] != backward_transfer['train_task'])]
        if num_init_tasks is not None:
            # discount the initialization phase
            backward_transfer = backward_transfer[backward_transfer['train_task'].astype(
                int) >= num_init_tasks]
        if reduce == 'mean':
            backward_transfer = backward_transfer['backward_transfer'].mean()
        return backward_transfer


# TODO: monitor metric related to data sharing accuracy, modules proning accuracy, federated learning stuff ect...
