'''
File: /metric.py
Project: utils
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import numpy as np
import matplotlib
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
            self.df = self.df[self.df['train_task'] >= num_init_tasks-1]
            # self.df = self.df[self.df['train_task'] >= num_init_tasks]

        self.num_init_tasks = num_init_tasks
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

    def compute_backward_transfer(self, reduce='mean'):
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

    def compute_catastrophic_forgetting(self, reduce='mean'):
        """
        catastrophic[t] = (accuracy of t at the end of train_task == t) - (accuracy of t at the end of lifetime0)

        High positive catastrophic is bad.
        "Negative catastrophic" is good: backward transfer.

        NOTE: Might be buggy with the joint initialization.
        """
        pre = self.df[(self.df['epoch'] == self.max_epoch) & (
            self.df['test_task'] == self.df['train_task'])]
        # get the pre for task before num_init_tasks by
        # setting the values to that when train_task == num_init_tasks -1
        pre_init = self.df[(self.df['epoch'] == self.max_epoch)
                           & (self.df['train_task'] == str(self.num_init_tasks - 1))]
        # combine pre and pre_init
        pre = pd.concat([pre, pre_init])
        # remove non-numeric test_task column (e.g., "avg")
        pre = pre[pd.to_numeric(
            pre['test_task'], errors='coerce').notnull()]
        # remove duplicate rows
        pre = pre.drop_duplicates(subset=['test_task'])

        max_task = str(self.get_max_tasks())
        post = self.df[(self.df['epoch'] == self.max_epoch) & (
            self.df['train_task'] == max_task)]

        # remove non-numeric
        post = post[pd.to_numeric(
            post['test_task'], errors='coerce').notnull()]

        catastrophic = pre.merge(
            post, on=['test_task'], suffixes=('_pre', '_post'))

        # catastrophic column is test_acc_pre - test_acc_post
        catastrophic['catastrophic'] = (catastrophic['test_acc_pre'] -
                                        catastrophic['test_acc_post'])

        if reduce == 'mean':
            # percentage
            catastrophic = catastrophic['catastrophic'].mean() * 100
        return catastrophic

    def compute_auc(self, mode='avg', tasks=None, metric='test_acc', num_init_tasks=4):
        """
        Compute the AUC based on mode and tasks specified.

        Parameters:
        - mode: Mode of AUC calculation, 'current' for learning curves of the same tasks, 'avg' for the averaged learning curve.
        - tasks: List of tasks to include in the calculation. If None, all tasks are included.
        - metric: The metric column name to use for AUC calculation. Default is 'test_acc'.

        Returns:
        - AUC value as a float.
        """
        # auc_values = []

        if tasks is None:
            # tasks = self.df['train_task'].unique()
            max_task = self.get_max_tasks()
            tasks = range(num_init_tasks, max_task+1)

        dfs = []
        for task in tasks:
            if mode == 'current':
                # Filter rows where train_task and test_task match the current task
                filtered_df = self.df[(self.df['train_task'] == str(
                    task)) & (self.df['test_task'] == str(task))]
            elif mode == 'avg':
                # Filter rows for the 'avg' test task across the specified tasks
                filtered_df = self.df[(self.df['train_task'] == str(
                    task)) & (self.df['test_task'] == 'avg')]
            else:
                raise ValueError("Invalid mode. Choose 'current' or 'avg'.")

            dfs.append(filtered_df)
            # Ensure the dataframe is sorted by epoch.
            # filtered_df = filtered_df.sort_values(by='epoch')

        df = pd.concat(dfs)
        agg_df = df.groupby('epoch').agg({metric: 'mean'}).reset_index()

        return np.trapz(agg_df[metric], agg_df['epoch'])


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


def get_magma_colors(total_elements):
    start = 0.2
    stop = 0.8
    cm_subsection = np.linspace(start, stop, total_elements)
    return [matplotlib.cm.plasma(x) for x in cm_subsection]


class DivergenceMetric:
    def __init__(self, save_dir, num_epochs=100) -> None:
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.file_path = os.path.join(self.save_dir, "sharing_record.csv")
        self.df = pd.read_csv(self.file_path)
        self.num_comm_rounds = self.df['communication_round'].max() + 1
        # self.df["epoch"] = self.df["task_id"] * \
        #     self.num_comm_rounds + self.df["communication_round"]

        self.compute_epochs()

    def compute_epochs(self):
        # Calculate the epoch increment for each communication round
        epoch_increment = self.num_epochs / self.num_comm_rounds

        # Apply the formula to calculate the epoch for each communication round and task_id
        self.df["epoch"] = self.df.apply(lambda row: (row['communication_round'] * epoch_increment) +
                                         (self.num_epochs * row['task_id']), axis=1)

        # Optionally, round the epoch to an integer if necessary
        self.df["epoch"] = self.df["epoch"].astype(int)
