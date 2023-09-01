'''
File: /utils.py
Project: utils
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import random
import os
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def create_dir_if_not_exist(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def viz_embedding(net, testloaders, name="test.png"):
    fig, ax = plt.subplots()
    X_out = []  # features
    y_out = []  # global labels
    y_task = []  # globallabel_task_id
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for task_id, testloader in testloaders.items():
            for X, y in testloader:
                X = X.to(net.device)
                X_encode = net.contrastive_embedding(X, task_id)
                X_out.append(X_encode.cpu())
                y_out.append(y.cpu())
                y_task.append(np.ones_like(y) * task_id)
    X_encode = np.concatenate(X_out, axis=0)
    Y = np.concatenate(y_out, axis=0)
    y_task = np.concatenate(y_task, axis=0)
    X_embedded = TSNE(n_components=2, random_state=0, init="pca",
                      n_jobs=-1).fit_transform(X_encode)

    # create an array same size as y and y_task where each element is {y}_{y_task} string
    y_task_str = np.array([str(Y[i]) + "_" + str(y_task[i])
                           for i in range(len(Y))])
    # plot X_embedded with color corresponding to y_task_str
    # different sns color palette
    # bigger plot size
    # sns.set(rc={'figure.figsize': (20, 10)})
    # sns.set_palette("tab20")
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_task_str,
                    ax=ax)
    plt.savefig(name)
    if was_training:
        net.train()

# save_eval takes into a net
