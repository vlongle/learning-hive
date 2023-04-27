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




import torch
def evaluate_data(func):
    def wrapper(data, agent, task_id):
        before = agent.evaluate(task_id)
        func(data, agent, task_id)
        after = agent.evaluate(task_id)
        return (after - before) / before
    return wrapper


"""
YTrueDataset: takes in (X, Y, task_id) and agent, and returns (X, Y_true)
"""


def get_ytrue_dataset(data, agent):
    class YTrueDataset(torch.utils.data.TensorDataset):
        def __getitem__(self, index):
            num_classes_per_task = agent.dataset.num_classes_per_task
            class_sequence = agent.dataset.class_sequence
            task_classes = class_sequence[task_id *
                                          num_classes_per_task: (task_id + 1) * num_classes_per_task]
            X, y, task_id = super().__getitem__(index)
            y_true = task_classes[y]
            return X, y_true

    return YTrueDataset(*data.tensors)


def remapping(data: torch.utils.data.TensorDataset, agent, task_id):
    """
    data.tensors: (X, Y, Y_true)
    return new_data.tensors: (X, Y_remap)
    """
    # dict mapping from Y_true to Y
    num_classes_per_task = agent.dataset.num_classes_per_task
    class_sequence = agent.dataset.class_sequence
    task_classes = class_sequence[task_id *
                                  num_classes_per_task: (task_id + 1) * num_classes_per_task]
    # Y_true should be in task_classes, map Y_true to index in task_classes
    # otherwise, map to -1

    class RemapTensorDataset(torch.utils.data.TensorDataset):
        def __getitem__(self, index):
            X, Y, Y_true = super().__getitem__(index)
            if Y_true in task_classes:
                Y = task_classes.index(Y_true)
            else:
                Y = -1
            return X, Y

    new_data = RemapTensorDataset(*data.tensors)
    return new_data


@evaluate_data
def global_utilize(data, agent, task_id):
    data = remapping(data, agent, task_id)
    agent.train(data)


@evaluate_data
def label_free_utilize(data, agent, task_id):
    pass


@evaluate_data
def pseudo_label_utilize(data, agent, task_id):
    pass
