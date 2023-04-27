'''
File: /datasets.py
Project: datasets
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import struct
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
import logging

import torchvision.transforms as transforms
SRC_DIR = os.path.join('src', 'shell')


class CustomTensorDataset(TensorDataset):
    # tensordataset but also apply transforms
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        tensors = super().__getitem__(index)
        x = tensors[0]
        # if self.transform, apply it on the first tensor
        if self.transform:
            x = self.transform(tensors[0])
        return tuple([x] + list(tensors[1:]))


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [x, self.transform(x)]


def get_transform(name, device="cuda"):
    if name == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=32, scale=(0.2, 1.), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])

    elif name == "fashionmnist":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(
                kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Lambda(
                lambda x: x + torch.randn(x.size()) * 0.05),
        ])

    elif name == "kmnist" or name == "mnist":
        # default augmentation (MNIST, KMNIST). Only affine and blur
        # since rotation might change the semantics of hand-writing digits
        # and Japanese letters.
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1),
                shear=0.1),
            transforms.GaussianBlur(
                kernel_size=3, sigma=(0.1, 2.0)),
        ])
    else:
        raise NotImplementedError(
            'Dataset {} not supported'.format(name))
    return train_transform


def get_custom_tensordataset(tensors, name, use_contrastive=False):
    """
    tensors: tuple of tensors
    """
    train_transform = None
    if use_contrastive:
        train_transform = TwoCropTransform(
            get_transform(name))

    return CustomTensorDataset(
        *tensors, transform=train_transform)


class SplitDataset():
    def __init__(self, num_tasks, num_classes, num_classes_per_task, with_replacement=False,
                 normalize=True, num_train_per_task=-1, num_val_per_task=-1, remap_labels=False,
                 num_init_tasks=None, labels=None, name=None, use_contrastive=False):
        self.num_classes = num_classes
        if not with_replacement:
            assert num_tasks <= num_classes // num_classes_per_task, 'Dataset does not support more than {} tasks'.format(
                num_classes // num_classes_per_task)
        self.num_tasks = num_tasks
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        if normalize:
            norm_val = X_train.max()
            X_train = X_train / norm_val
            X_test = X_test / norm_val
            X_val = X_val / norm_val

        self.trainset = []
        self.valset = []
        self.testset = []
        self.features = []

        self.max_batch_size = 0
        if labels is None:
            if not with_replacement:
                labels = np.random.permutation(num_classes)
            else:
                labels = np.array([np.random.choice(
                    num_classes, num_classes_per_task, replace=False) for t in range(self.num_tasks)])
                labels = labels.reshape(-1)

        # if num_init_tasks is not None and with_replacement:
        #     # NOTE: HACK: remove this.
        #     # make sure that the first num_init_tasks * num_classes_per_task classes are ALL DISTINCT!
        #     labels[:num_init_tasks*num_classes_per_task] = np.random.choice(
        #         num_classes, num_init_tasks*num_classes_per_task, replace=False)

        self.class_sequence = labels
        logging.info(f"Class sequence: {self.class_sequence}")
        for task_id in range(self.num_tasks):

            Xb_train_t, yb_train_t, Xb_val_t, yb_val_t, Xb_test_t, yb_test_t = \
                self.split_data(X_train, y_train, X_val, y_val, X_test, y_test, labels[np.arange(
                    task_id*num_classes_per_task, (task_id+1)*num_classes_per_task)], remap_labels=remap_labels)
            if num_train_per_task != -1:
                Xb_train_t = Xb_train_t[:num_train_per_task]
                yb_train_t = yb_train_t[:num_train_per_task]
            if num_val_per_task != -1:
                Xb_val_t = Xb_val_t[:num_val_per_task]
                yb_val_t = yb_val_t[:num_val_per_task]
            logging.info(f"task {task_id} :{Xb_train_t.shape}")

            yb_train_t = torch.from_numpy(yb_train_t).long().squeeze()
            yb_val_t = torch.from_numpy(yb_val_t).long().squeeze()
            yb_test_t = torch.from_numpy(yb_test_t).long().squeeze()

            Xb_train_t = torch.from_numpy(Xb_train_t).float()
            Xb_val_t = torch.from_numpy(Xb_val_t).float()
            Xb_test_t = torch.from_numpy(Xb_test_t).float()

            self.trainset.append(get_custom_tensordataset(
                [Xb_train_t, yb_train_t], name=name, use_contrastive=use_contrastive))
            self.valset.append(TensorDataset(Xb_val_t, yb_val_t))
            self.testset.append(TensorDataset(Xb_test_t, yb_test_t))

            self.features.append(Xb_train_t.shape[2])

        self.max_batch_size = 128
        if remap_labels:
            self.num_classes = num_classes_per_task

    def split_data(self, X_train, y_train, X_val, y_val, X_test, y_test, labels, remap_labels=True):
        Xb_train = X_train[np.isin(y_train, labels)]
        yb_train = y_train.copy()
        if remap_labels:
            for i in range(len(labels)):
                yb_train[y_train == labels[i]] = i
        yb_train = yb_train[np.isin(y_train, labels)]

        Xb_val = X_val[np.isin(y_val, labels)]
        yb_val = y_val.copy()
        if remap_labels:
            for i in range(len(labels)):
                yb_val[y_val == labels[i]] = i
        yb_val = yb_val[np.isin(y_val, labels)]

        Xb_test = X_test[np.isin(y_test, labels)]
        yb_test = y_test.copy()
        if remap_labels:
            for i in range(len(labels)):
                yb_test[y_test == labels[i]] = i
        yb_test = yb_test[np.isin(y_test, labels)]

        return Xb_train, yb_train, Xb_val, yb_val, Xb_test, yb_test

    def load_data(self):
        raise NotImplementedError('This method must be dataset specific')


class MNIST(SplitDataset):
    def __init__(self, num_tasks=5, num_classes_per_task=2, with_replacement=False,
                 num_train_per_task=-1, num_val_per_task=-1, remap_labels=False, num_init_tasks=None,
                 labels=None, use_contrastive=False, name=None):
        if name is None:
            name = "mnist"
        super().__init__(num_tasks, num_classes=10, num_classes_per_task=num_classes_per_task,
                         with_replacement=with_replacement, num_train_per_task=num_train_per_task, num_val_per_task=num_val_per_task, remap_labels=remap_labels,
                         num_init_tasks=num_init_tasks,
                         labels=labels, name=name, use_contrastive=use_contrastive)
        self.name = name

    def load_data(self):
        with open(os.path.join(SRC_DIR, 'datasets/mnist/train-labels.idx1-ubyte'), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_train = np.fromfile(flbl, dtype=np.int8)

        with open(os.path.join(SRC_DIR, 'datasets/mnist/train-images.idx3-ubyte'), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_train = np.fromfile(fimg, dtype=np.uint8).reshape(
                len(y_train), rows, cols)

        with open(os.path.join(SRC_DIR, 'datasets/mnist/t10k-labels.idx1-ubyte'), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_test = np.fromfile(flbl, dtype=np.int8)

        with open(os.path.join(SRC_DIR, 'datasets/mnist/t10k-images.idx3-ubyte'), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_test = np.fromfile(fimg, dtype=np.uint8).reshape(
                len(y_test), rows, cols)

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]

        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        return X_train, y_train, X_val, y_val, X_test, y_test


class FashionMNIST(MNIST):
    '''
    Since the structure is identical to MNIST, we can simply change
    the data directory, and maintain the rest of the MNIST loaders
    '''

    def __init__(self, num_tasks=5, num_classes_per_task=2, with_replacement=False,
                 num_train_per_task=-1, num_val_per_task=-1, remap_labels=False,
                 num_init_tasks=None, labels=None, use_contrastive=False):
        super().__init__(num_tasks, num_classes_per_task=num_classes_per_task,
                         with_replacement=with_replacement, num_train_per_task=num_train_per_task,
                         num_val_per_task=num_val_per_task, remap_labels=remap_labels,
                         num_init_tasks=num_init_tasks, labels=labels, name='fashionmnist',
                         use_contrastive=use_contrastive)
        self.name = 'fashionmnist'

    def load_data(self):
        with open(os.path.join(SRC_DIR, 'datasets/fashion/train-labels.idx1-ubyte'), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_train = np.fromfile(flbl, dtype=np.int8)

        with open(os.path.join(SRC_DIR, 'datasets/fashion/train-images.idx3-ubyte'), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_train = np.fromfile(fimg, dtype=np.uint8).reshape(
                len(y_train), rows, cols)

        with open(os.path.join(SRC_DIR, 'datasets/fashion/t10k-labels.idx1-ubyte'), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_test = np.fromfile(flbl, dtype=np.int8)

        with open(os.path.join(SRC_DIR, 'datasets/fashion/t10k-images.idx3-ubyte'), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_test = np.fromfile(fimg, dtype=np.uint8).reshape(
                len(y_test), rows, cols)

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]

        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        return X_train, y_train, X_val, y_val, X_test, y_test


class KMNIST(MNIST):
    def __init__(self, num_tasks=5, num_classes_per_task=2, with_replacement=False,
                 num_train_per_task=-1, num_val_per_task=-1, remap_labels=False,
                 num_init_tasks=None, labels=None, use_contrastive=False):
        super().__init__(num_tasks, num_classes_per_task=num_classes_per_task,
                         with_replacement=with_replacement, num_train_per_task=num_train_per_task,
                         num_val_per_task=num_val_per_task, remap_labels=remap_labels,
                         num_init_tasks=num_init_tasks, labels=labels, name='kmnist',
                         use_contrastive=use_contrastive)
        self.name = 'kmnist'

    def load_data(self):
        def load(f):
            return np.load(f)['arr_0']

        # Load the data
        X_train = load(os.path.join(
            SRC_DIR, 'datasets/kmnist/kmnist-train-imgs.npz'))
        X_test = load(os.path.join(
            SRC_DIR, 'datasets/kmnist/kmnist-test-imgs.npz'))
        y_train = load(os.path.join(
            SRC_DIR, 'datasets/kmnist/kmnist-train-labels.npz'))
        y_test = load(os.path.join(
            SRC_DIR, 'datasets/kmnist/kmnist-test-labels.npz'))

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]

        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)
            X_test = np.expand_dims(X_test, axis=1)

        return X_train, y_train, X_val, y_val, X_test, y_test


class CIFAR100(SplitDataset):
    def __init__(self, num_tasks=20, num_classes_per_task=5, with_replacement=False,
                 num_train_per_task=-1, num_val_per_task=-1, remap_labels=False,
                 num_init_tasks=None, labels=None, use_contrastive=False):
        super().__init__(num_tasks, num_classes=100, num_classes_per_task=num_classes_per_task,
                         with_replacement=with_replacement, num_train_per_task=num_train_per_task,
                         num_val_per_task=num_val_per_task, remap_labels=remap_labels,
                         num_init_tasks=num_init_tasks, labels=labels, name='cifar100',
                         use_contrastive=use_contrastive)
        self.name = 'cifar100'

    def load_data(self):
        with open(os.path.join(SRC_DIR, 'datasets/cifar-100/train'), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        y_train = np.array(data_dict[b'fine_labels'])
        X_train = data_dict[b'data'].reshape(-1, 3, 32, 32)
        s = X_train.shape

        with open(os.path.join(SRC_DIR, 'datasets/cifar-100/test'), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        y_test = np.array(data_dict[b'fine_labels'])
        X_test = data_dict[b'data'].reshape(-1, 3, 32, 32)

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]
        return X_train, y_train, X_val, y_val, X_test, y_test


def get_dataset(**kwargs):
    dataset_name = kwargs.get('dataset_name', 'mnist')
    kwargs.pop('dataset_name', None)
    if dataset_name == 'mnist':
        return MNIST(**kwargs)
    elif dataset_name == 'fashionmnist':
        return FashionMNIST(**kwargs)
    elif dataset_name == 'kmnist':
        return KMNIST(**kwargs)
    elif dataset_name == 'cifar100':
        return CIFAR100(**kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
