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




import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
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


def get_global_label(local_y: int, task_id: int, source_class_sequence, num_classes_per_task) -> int:
    # convert local_y to global_y
    task_classes = source_class_sequence[task_id *
                                         num_classes_per_task: (task_id + 1) * num_classes_per_task]
    global_y = task_classes[local_y]
    return global_y


def get_global_labels(local_ys, task_ids, source_class_sequence, num_classes_per_task):
    if isinstance(task_ids, int):
        task_ids = [task_ids] * len(local_ys)
    global_ys = []
    for local_y, task_id in zip(local_ys, task_ids):
        global_y = get_global_label(
            local_y, task_id, source_class_sequence, num_classes_per_task)
        global_ys.append(global_y)
    return torch.tensor(global_ys)


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
            global_y = get_global_label(
                local_y, task_id, source_class_sequence, num_classes_per_task)
            return X, global_y

    return GlobalLabelsDataset(data)


def get_local_label_for_task(global_y, target_task_id, target_class_sequence, num_classes_per_task):
    """
    Get the local label for the target task. If global_y is not in the target task, return -1.
    """
    task_classes = list(target_class_sequence[target_task_id * num_classes_per_task: (
        target_task_id + 1) * num_classes_per_task])
    local_y = task_classes.index(global_y) if global_y in task_classes else -1
    return local_y


def get_local_labels_for_task(global_ys, target_task_id, target_class_sequence, num_classes_per_task):
    local_ys = []
    for global_y in global_ys:
        local_y = get_local_label_for_task(
            global_y, target_task_id, target_class_sequence, num_classes_per_task)
        local_ys.append(local_y)
    return torch.tensor(local_ys)


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
            local_y = get_local_label_for_task(
                global_y, target_task_id, target_class_sequence, num_classes_per_task)
            return X, local_y

    return LocalLabelsForTaskDataset(data)

# NOTE: just find the first task, a bit problematic


def get_local_label(global_y, target_class_sequence, num_classes_per_task):
    num_tasks = len(target_class_sequence) // num_classes_per_task
    local_y = -1
    local_task_id = -1
    for task_id in range(num_tasks):
        temp_local_y = get_local_label_for_task(
            global_y, task_id, target_class_sequence, num_classes_per_task)
        if temp_local_y != -1:
            local_y = temp_local_y
            local_task_id = task_id
            break
    return local_y, local_task_id


def get_local_labels(global_ys, target_class_sequence, num_classes_per_task):
    local_ys, local_task_ids = [], []
    for global_y in global_ys:
        local_y, local_task_id = get_local_label(
            global_y, target_class_sequence, num_classes_per_task)
        local_ys.append(local_y)
        local_task_ids.append(local_task_id)
    return torch.tensor(local_ys), torch.tensor(local_task_ids)


# def get_all_local_labels(global_ys, target_class_sequence, num_classes_per_task):
#     local_ys, local_task_ids = [], []
#     for global_y in global_ys:
#         all_local_y, all_local_task_id = [], []
#         for task_id, class_sequence in enumerate(target_class_sequence):
#             if global_y in class_sequence:
#                 local_y = class_sequence.index(global_y)
#                 all_local_y.append(local_y)
#                 all_local_task_id.append(task_id)
#         local_ys.append(all_local_y)
#         local_task_ids.append(all_local_task_id)
#     return local_ys, local_task_ids

def get_all_local_labels(global_ys, target_class_sequence, num_classes_per_task):
    num_tasks = len(target_class_sequence) // num_classes_per_task
    all_local_ys, all_local_task_ids = [], []

    for global_y in global_ys:
        local_ys_for_global_y = []
        task_ids_for_global_y = []
        for task_id in range(num_tasks):
            local_y = get_local_label_for_task(
                global_y, task_id, target_class_sequence, num_classes_per_task)
            if local_y != -1:
                local_ys_for_global_y.append(local_y)
                task_ids_for_global_y.append(task_id)

        all_local_ys.append(local_ys_for_global_y)
        all_local_task_ids.append(task_ids_for_global_y)

    return all_local_ys, all_local_task_ids


def remap_to_local_labels(data, class_sequence, num_classes_per_task):
    """Remap global labels to the agent's local labels and determine the corresponding local task."""

    class LocalLabelsDataset(torch.utils.data.TensorDataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            X, global_y = self.dataset[index]
            local_y, local_task_id = get_local_label(
                global_y, class_sequence, num_classes_per_task)
            return X, local_y, local_task_id

    return LocalLabelsDataset(data)


def utilize_global_labels(data, source_class_sequence, target_class_sequence, num_classes_per_task):
    """
    Convert data with source class sequence to target class sequence.
    Input: data (X, y_source, task_source_id)
    Output: new_data (X_target, y_target, task_target_id)
    """

    # Convert the dataset with local task-specific labels to a dataset with global labels
    global_label_data = get_global_labels_dataset(
        data, source_class_sequence, num_classes_per_task)

    # Remap global labels to the target's local labels and determine the corresponding local task
    local_label_data = remap_to_local_labels(
        global_label_data, target_class_sequence, num_classes_per_task)

    return local_label_data


class RandomFlippedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, flip_probability, num_classes,
                 jerk=False):
        """
        dataset: Input dataset with data points (X, y)
        flip_probability: Probability of flipping the true label of a data point
        num_classes: Total number of possible classes (to choose a random label if flipping)
        """
        self.dataset = dataset
        self.flip_probability = flip_probability
        self.num_classes = num_classes
        self.jerk = jerk

    def __getitem__(self, index):
        X, y, task_id = self.dataset[index]

        # With a probability of flip_probability, change the label to a random one
        # if jerk is True, then the new label must be different from the old one
        if random.random() < self.flip_probability:
            if self.jerk:
                # TODO
                possible_labels = [label for label in range(
                    self.num_classes) if label != y]
                # Choose a random label from the generated list
                y = random.choice(possible_labels)
            else:
                y = random.randint(0, self.num_classes - 1)

        return X, y, task_id

    def __len__(self):
        return len(self.dataset)


"""
Pseudo-labeling using source consistency
and then applies confidence

1. Compute Predictions: For each target task, use its decoder to predict labels for X_source.
2. Determine Consistency: For each target task's predictions, determine the mode (most common label). The consistency score is the fraction of the predicted labels that match the mode.
3. Rank Decoders: Rank the decoders based on the consistency score. The higher the consistency score, the more likely the class in X_source was present during the training of that task.

"""
# def rank_and_pseudo_label(agent, X_source):
#     task_scores = []

#     for task_id in range(agent.num_tasks):
#         outputs = agent.predict(X_source, task_id)
#         _, predicted_class = outputs.max(dim=1)

#         # Find the mode of the predicted_class
#         mode_label = torch.mode(predicted_class).values.item()
#         mode_label_count = (predicted_class == mode_label).sum().item()

#         # Consistency score: how frequently the mode occurs.
#         consistency_score = mode_label_count / len(predicted_class)
#         task_scores.append((task_id, consistency_score, mode_label))

#     # Sort tasks by their consistency score
#     sorted_tasks = sorted(task_scores, key=lambda x: x[1], reverse=True)

#     # Most consistent task details
#     best_task_id, _, best_task_mode_label = sorted_tasks[0]

#     # For confidence, get the softmax scores from the best task's predictions
#     outputs_best_task = agent.predict(X_source, best_task_id)
#     confidences = torch.nn.functional.softmax(outputs_best_task, dim=1)

#     # Get the confidence values of the best task mode label
#     confidences_best_label = confidences[:, best_task_mode_label]

#     # Generate pseudo-labels: all images get the mode label of the best task's predictions
#     pseudo_labels = torch.full((len(X_source),), best_task_mode_label, dtype=torch.long)

#     return pseudo_labels, confidences_best_label, best_task_id


@torch.inference_mode()
def rank_and_pseudo_label(agent, X_source):
    '''
    A bit problematic.
    '''
    total_tasks = agent.dataset.num_tasks
    num_classes_per_task = agent.dataset.num_classes_per_task
    label_confidence_matrix = []
    label_count_matrix = []
    was_training = agent.net.training

    # Set network to eval mode
    agent.net.eval()

    # Accumulate results from all task decoders
    for task_id in range(total_tasks):
        predictions = agent.net(X_source.to(agent.net.device), task_id)

        # Get the predicted labels and their confidences
        predicted_labels = torch.argmax(predictions, dim=1).cpu()
        confidences = torch.max(F.softmax(predictions, dim=1), dim=1)[0].cpu()
        print("task_id", task_id)
        print("predicted_labels", predicted_labels)
        print('\n\n')

        # Accumulate label counts
        label_counts = torch.bincount(
            predicted_labels, minlength=num_classes_per_task)
        label_count_matrix.append(label_counts)

        # Sum up the confidences for each unique label
        summed_confidences = torch.zeros(num_classes_per_task)
        for i in range(num_classes_per_task):
            summed_confidences[i] = confidences[predicted_labels == i].sum()
        label_confidence_matrix.append(summed_confidences)

    # Convert lists to tensors for further operations
    label_count_matrix = torch.stack(label_count_matrix)
    label_confidence_matrix = torch.stack(label_confidence_matrix)

    # Get average confidences
    avg_confidences = label_confidence_matrix / \
        (label_count_matrix + 1e-10)  # Avoid division by zero

    # # Rank task decoders based on mode consistency and confidence
    total_scores = label_count_matrix + avg_confidences
    # # best_task_id = torch.argmax(total_scores.sum(dim=1))
    # total_scores = avg_confidences

    max_scores, max_indices = total_scores.max(dim=1)
    best_task_id = torch.argmax(max_scores)

    best_label = torch.argmax(total_scores[best_task_id])

    # Debugging information
    debug_info = {
        "predicted_label_matrix": label_count_matrix,
        "confidences_matrix": avg_confidences,
        "total_scores": total_scores,
    }

    if was_training:
        agent.net.train()

    return best_task_id, best_label, debug_info


def pseudo_label(agent, X_source, threshold=0.9):
    """
    Returns pseudo-labeled data based on the most consistent decoder of the agent.

    Parameters:
    - agent: The agent used for predictions.
    - X_source: The source data to be pseudo-labeled.
    - threshold: The minimum confidence required to keep a pseudo-label.

    Returns:
    - X_ret: Source data that passed the confidence threshold.
    - y_target: Pseudo-labels for X_ret.
    - task_target_id: The id of the most consistent decoder.
    """

    # Using the previous function to rank tasks and obtain pseudo-labels
    task_rankings, task_labels, confidences = rank_and_pseudo_label(
        agent, X_source)

    # Identifying the top-ranked task and its associated labels and confidences
    top_task_id = task_rankings[0]
    top_task_labels = task_labels[top_task_id]
    top_task_confidences = confidences[top_task_id]

    # Applying confidence thresholding
    mask = top_task_confidences > threshold
    X_ret = X_source[mask]
    y_target = top_task_labels[mask]
    task_target_id = torch.full(y_target.shape, top_task_id, dtype=torch.int64)

    return X_ret, y_target, task_target_id


# @evaluate_data
# def label_free_utilize(data, agent, task_id):
#     """
#     Data should be a mono dataset
#     """
#     pass


# @evaluate_data
# def pseudo_label_utilize(data, agent, task_id):
#     """
#     Data should be a mono dataset
#     """
#     pass


def filter_dataset_by_label(dataset, target_label):
    """Filter tensor dataset to only contain target_label. Return a new dataset."""
    labels = dataset.tensors[1]
    mask = labels == target_label
    return torch.utils.data.TensorDataset(*[t[mask] for t in dataset.tensors])


def compute_tasks_sim(task1, task2):
    assert len(task1) == len(
        task2), f"len(task1)={len(task1)} != len(task2)={len(task2)}"
    union = len(set(task1) | set(task2))
    intersection = len(set(task1) & set(task2))
    return intersection / union if union > 0 else 0


def send_labels(Y_globals, task, add_offset=None,
                to_tasks=True):
    '''
    Y_globals is a list of global labels, and task is a list of classes in the current task.
    For example,
    `Y_globals = [0, 0, 1, 0, 1, 4, 9, 8, 7]` and `task = [1,4]`

    If `add_offset` is not None, then the global labels are offset by `add_offset` before sending.

    If `to_tasks=True` then `add_offset` must not be None. In this case, the labels in the task are converted
    to the local labels, and the rest are added with the offset to distinguish them from the local labels. For
    example, `to_tasks=True` and `add_offsets=100` will yield
    ```
    [100, 100, 0, 100, 0, 1, 109, 108, 107]. Note that 1 and 4 are converted to 0 and 1, and the rest are added
    with 100.
    ```
    '''
    if to_tasks and add_offset is None:
        raise ValueError(
            "If `to_tasks=True` then `add_offset` must not be None.")

    if to_tasks:
        Y_globals = [y if y in task else y + add_offset for y in Y_globals]
    elif add_offset is not None:
        Y_globals = [y + add_offset for y in Y_globals]
    return Y_globals
