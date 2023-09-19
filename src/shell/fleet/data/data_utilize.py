'''
File: /send_utilize.py
Project: data
Created Date: Sunday April 23rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

"""
Given a batch of data (X, Y, Y_true) and a learner, evaluate the usefulness of the data
for task `t` of the learner.
1. Global labels: use Y_true, and match against the class_sequence of the learner. Remap Y to 
the learner's Y. Then fit.
2. Label-free representation. Fit contrastive / cross entropy with (X, Y) as it is. Expect CE to be
very bad.
3. Pseudo-label: use the learner's labels to get Y_pred and use that to fit. 
"""

# make a decorator function that first evaluate on task_id,
# call the fit function of the learner, and then evaluate on task_id again
# return the difference in performance

import logging
import numpy as np
import torch
def evaluate_data(func):
    def wrapper(data, agent, task_id):
        cur_task = len(agent.agent.replay_buffers) - 1
        testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=256,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(agent.dataset.testset[:(cur_task+1)])}
        before_loss, before_acc = agent.agent.evaluate(
            testloaders, eval_no_update=False)
        func(data, agent, task_id)
        after_loss, after_acc = agent.agent.evaluate(
            testloaders, eval_no_update=False)
        # diff = (after[task_id] - before[task_id]) / before[task_id]
        diff = {t: (after_acc.get(t, np.inf) - before_acc[t]) / before_acc[t]
                for t in before_acc}
        return diff
    return wrapper


def get_global_label(local_y, task_id, source_class_sequence, num_classes_per_task):
    # convert local_y to global_y
    task_classes = source_class_sequence[task_id * num_classes_per_task: (task_id + 1) * num_classes_per_task]
    global_y = task_classes[local_y]
    return global_y

def get_global_labels_dataset(data, source_class_sequence, num_classes_per_task):
    """
    Convert local task-specific labels to global labels.
    The class sequence provide the global labels in the task.
    Input data: (X, Y, task_id)
    Output data: (X, global_Y)
    """
    class GlobalLabelsDataset(torch.utils.data.TensorDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            X, local_y, task_id = self.dataset[index]
            global_y = get_global_label(local_y, task_id, source_class_sequence, num_classes_per_task)
            return X, global_y

    return GlobalLabelsDataset(data)


def get_local_label_for_task(global_y, target_task_id, target_class_sequence, num_classes_per_task):
    """
    Get the local label for the target task. If global_y is not in the target task, return -1.
    """
    task_classes = list(target_class_sequence[target_task_id * num_classes_per_task: (target_task_id + 1) * num_classes_per_task])
    local_y = task_classes.index(global_y) if global_y in task_classes else -1
    return local_y

def remap_to_task_local_labels(data, target_class_sequence, num_classes_per_task, target_task_id):
    """
    Remap global labels to the agent's local task-specific labels.
    Map unseen classes to -1
    """

    class LocalLabelsForTaskDataset(torch.utils.data.TensorDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            X, global_y = self.dataset[index]
            local_y = get_local_label_for_task(global_y, target_task_id, target_class_sequence, num_classes_per_task)
            return X, local_y

    return LocalLabelsForTaskDataset(data)


def get_local_label(global_y, target_class_sequence, num_classes_per_task):
    num_tasks = len(target_class_sequence) // num_classes_per_task
    local_y = -1
    local_task_id = -1
    for task_id in range(num_tasks):
        temp_local_y = get_local_label_for_task(global_y, task_id, target_class_sequence, num_classes_per_task)
        if temp_local_y != -1:
            local_y = temp_local_y
            local_task_id = task_id
            break
    return local_y, local_task_id


def remap_to_local_labels(data, class_sequence, num_classes_per_task):
    """Remap global labels to the agent's local labels and determine the corresponding local task."""

    class LocalLabelsDataset(torch.utils.data.TensorDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            X, global_y = self.dataset[index]
            local_y, local_task_id = get_local_label(global_y, class_sequence, num_classes_per_task)
            return X, local_y, local_task_id

    return LocalLabelsDataset(data)


def utilize_global_labels(data, source_class_sequence, target_class_sequence, num_classes_per_task):
    """
    Convert data with source class sequence to target class sequence.
    Input: data (X, y_source, task_source_id)
    Output: new_data (X_target, y_target, task_target_id)
    """

    # Convert the dataset with local task-specific labels to a dataset with global labels
    global_label_data = get_global_labels_dataset(data, source_class_sequence, num_classes_per_task)
    
    # Remap global labels to the target's local labels and determine the corresponding local task
    local_label_data = remap_to_local_labels(global_label_data, target_class_sequence, num_classes_per_task)
    
    return local_label_data


@evaluate_data
def label_free_utilize(data, agent, task_id):
    """
    Data should be a mono dataset
    """
    pass


@evaluate_data
def pseudo_label_utilize(data, agent, task_id):
    """
    Data should be a mono dataset
    """
    pass



def filter_dataset_by_label(dataset, target_label):
    """Filter tensor dataset to only contain target_label. Return a new dataset."""
    labels = dataset.tensors[1]
    mask = labels == target_label
    return torch.utils.data.TensorDataset(*[t[mask] for t in dataset.tensors])