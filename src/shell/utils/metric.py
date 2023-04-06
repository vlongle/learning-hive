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

    def __init__(self, save_dir, num_init_tasks=None):
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, 'record.csv')
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
        else:
            logging.critical(f"File {self.file_path} does not exist")
            exit(1)

        # HACK: for grad, we also log these immediate retraining during sync
        # remove self.df['train_task'] that cannot be converted to numeric
        self.df = self.df[pd.to_numeric(
            self.df['train_task'], errors='coerce').notnull()]
        # convert train_task column to numeric
        self.df['train_task'] = pd.to_numeric(self.df['train_task'])

        if num_init_tasks is not None:
            # throw away all the data before num_init_tasks
            # because it was before any training happens.
            # NOTE: num_init_tasks-1 is actually more correct for avg_accuracy
            # computation, for final_acc it doesn't matter. We're using
            # init_num_epochs != num_epochs, and so for hacky reasons.
            # keeping, num_init_tasks sol
            # self.df = self.df[self.df['train_task'] >= num_init_tasks-1]
            self.df = self.df[self.df['train_task'] >= num_init_tasks]

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

    def get_max_tasks(self):
        # self.df["train_task"] is a string, convert
        # to numeric (ignore non-numeric values) and take the max
        max_task = pd.to_numeric(self.df["train_task"], errors="coerce").max()
        return max_task

    def compute_final_accuracy(self, reduce='mean'):
        """
        Compute the final accuracy over all tasks at the end of lifetime.
        """
        max_task = str(self.get_max_tasks())
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
        if reduce == 'mean':
            backward_transfer = backward_transfer['backward_transfer'].mean()
        return backward_transfer


def task_similarity(classes_sequence_list, num_tasks, num_classes_per_task):
    """
    Create a dataframe with the following columns:
        - task_id: task id
        - agent_1_id: agent id
        - agent_2_id: agent id
        - similarity: number of classes in common between agent_1 and agent_2 at task_id
        - running_similarity: running similarity between agent_1 and agent_2 up to task_id
    """
    for classes_sequence in classes_sequence_list:
        assert len(classes_sequence) == num_tasks * num_classes_per_task

    num_agents = len(classes_sequence_list)
    df = pd.DataFrame(columns=['task_id', 'agent_1_id', 'agent_2_id', 'similarity',
                               'running_similarity'])
    for task in range(num_tasks):
        for agent_1 in range(num_agents):
            for agent_2 in range(agent_1 + 1, num_agents):
                classes_1 = classes_sequence_list[agent_1][
                    task * num_classes_per_task:(task + 1) * num_classes_per_task]
                classes_2 = classes_sequence_list[agent_2][
                    task * num_classes_per_task:(task + 1) * num_classes_per_task]
                similarity = len(set(classes_1) & set(classes_2))
                if task == 0:
                    running_similarity = 0
                else:
                    running_similarity = df[(df['task_id'] == task - 1) & (
                        df['agent_1_id'] == agent_1) & (df['agent_2_id'] == agent_2)]['running_similarity'].item()
                row = {
                    'task_id': task,
                    'agent_1_id': agent_1,
                    'agent_2_id': agent_2,
                    'similarity': similarity,
                    'running_similarity': running_similarity + similarity
                }
                df = pd.concat([df, pd.DataFrame(row, index=[0])])
    return df

# TODO: monitor metric related to data sharing accuracy, modules proning accuracy, federated learning stuff ect...
