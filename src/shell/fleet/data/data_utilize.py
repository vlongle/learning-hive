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


"""
YTrueDataset: takes in (X, Y, task_id) and agent, and returns (X, Y_true)
"""


def get_ytrue_dataset(data, class_sequence, num_classes_per_task):
    class YTrueDataset(torch.utils.data.TensorDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            X, y, task_id = self.dataset[index]
            task_classes = class_sequence[task_id *
                                          num_classes_per_task: (task_id + 1) * num_classes_per_task]
            y_true = task_classes[y]
            return X, y_true

    return YTrueDataset(data)


def remapping(data: torch.utils.data.TensorDataset, class_sequence, num_classes_per_task, task_id):
    """
    data.tensors: (X, Y_true)
    return new_data.tensors: (X, Y_remap)
    """
    # dict mapping from Y_true to Y
    task_classes = list(class_sequence[task_id *
                                       num_classes_per_task: (task_id + 1) * num_classes_per_task])
    # Y_true should be in task_classes, map Y_true to index in task_classes
    # otherwise, map to -1

    class RemapTensorDataset(torch.utils.data.TensorDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            X, Y_true = self.dataset[index]
            if Y_true in task_classes:
                Y = task_classes.index(Y_true)
            else:
                Y = -1
            return X, Y

    new_data = RemapTensorDataset(data)
    return new_data


@evaluate_data
def global_utilize(monodata_true, agent, task_id):
    """
    Data should be a mono dataset
    """
    monodata_remap = remapping(
        monodata_true, agent.dataset.class_sequence, agent.dataset.num_classes_per_task, task_id)
    y = monodata_remap[0][1]
    if y == -1:
        logging.debug("Unseen class")
        return False
    logging.debug("Seen class")
    return True


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


def get_mono_dataset(dataset, target_y):
    # dataset.tensors[1] contains the class information
    # filter dataset to only contain target_y. Return
    # a new dataset
    Y = dataset.tensors[1]
    mask = Y == target_y
    return torch.utils.data.TensorDataset(*[t[mask] for t in dataset.tensors])
