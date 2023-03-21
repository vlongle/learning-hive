'''
File: /base_learning_classes.py
Project: learners
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch
import torch.nn as nn
import os
from itertools import zip_longest
from shell.utils.utils import create_dir_if_not_exist
from shell.utils.record import Record
from torch.utils.tensorboard import SummaryWriter
import logging


class Learner():
    def __init__(self, net, save_dir='./tmp/results/', improvement_threshold=0.05):
        self.net = net
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.improvement_threshold = improvement_threshold
        self.T = 0
        self.observed_tasks = set()

        self.save_dir = create_dir_if_not_exist(save_dir)
        self.record = Record(os.path.join(self.save_dir, "record.csv"))
        self.writer = SummaryWriter(
            log_dir=create_dir_if_not_exist(os.path.join(self.save_dir, "tensorboard/")))
        self.init_trainloaders = None

    def train(self, *args, **kwargs):
        raise NotImplementedError('Training loop is algorithm specific')

    def init_train(self, trainloader, task_id, num_epochs, save_freq=1, testloaders=None):
        if self.init_trainloaders is None:
            self.init_trainloaders = {}
        self.init_trainloaders[task_id] = trainloader
        if len(self.init_trainloaders) == self.net.num_init_tasks:
            iter_cnt = 0
            for i in range(num_epochs):
                for XY_all in zip_longest(*self.init_trainloaders.values()):
                    for task, XY in zip(self.init_trainloaders.keys(), XY_all):
                        if XY is not None:
                            X, Y = XY
                            X = X.to(self.net.device, non_blocking=True)
                            Y = Y.to(self.net.device, non_blocking=True)
                            self.gradient_step(X, Y, task)
                            iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    self.save_data(i + 1, task_id, testloaders)

            self.save_data(num_epochs + 1, task_id,
                           testloaders, final_save=True)
            for task, loader in self.init_trainloaders.items():
                self.update_multitask_cost(loader, task)
        else:
            self.save_data(0, task_id,
                           testloaders, final_save=True)

    def evaluate(self, testloaders):
        was_training = self.net.training
        prev_reduction = self.loss.reduction
        self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        self.net.eval()
        with torch.no_grad():
            self.test_loss = {}
            self.test_acc = {}
            for task, loader in testloaders.items():
                l = 0.
                a = 0.
                n = len(loader.dataset)
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task)
                    l += self.loss(Y_hat, Y).item()
                    a += (Y_hat.argmax(dim=1) == Y).sum().item()
                self.test_loss[task] = l / n
                self.test_acc[task] = a / n

        self.loss.reduction = prev_reduction
        if was_training:
            self.net.train()

    def gradient_step(self, X, Y, task_id):
        Y_hat = self.net(X, task_id=task_id)
        l = self.loss(Y_hat, Y)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

    def save_data(self, epoch, task_id, testloaders, final_save=False):
        self.evaluate(testloaders)
        task_results_dir = os.path.join(
            self.save_dir, 'task_{}'.format(task_id))
        os.makedirs(task_results_dir, exist_ok=True)
        line = 'epochs: {}, training task: {}'.format(epoch, task_id)
        logging.info(line)
        for task in self.test_loss:
            # if self.test_acc[task] is a tuple, take the max
            if isinstance(self.test_acc[task], tuple):
                self.test_acc[task] = max(self.test_acc[task])
            if isinstance(self.test_loss[task], tuple):
                self.test_loss[task] = max(self.test_loss[task])

        if "avg" not in self.test_acc:
            self.test_acc["avg"] = sum(
                self.test_acc.values()) / len(self.test_acc)
        if "avg" not in self.test_loss:
            self.test_loss["avg"] = sum(
                self.test_loss.values()) / len(self.test_loss)

        for task in self.test_loss:
            line = '\ttask: {}\tloss: {:.3f}\tacc: {:.3f}'.format(
                task, self.test_loss[task], self.test_acc[task])
            logging.info(line)
            self.record.write(
                {
                    'train_task': task_id,
                    'test_task': task,
                    'test_acc': self.test_acc[task],
                    'test_loss': self.test_loss[task],
                    'epoch': epoch,
                }
            )

        if final_save:
            path = os.path.join(
                task_results_dir, 'checkpoint.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'observed_tasks': self.observed_tasks,
            }, path)
            self.record.save()

    def update_multitask_cost(self, loader, task_id):
        raise NotImplementedError(
            'Update update_multitask is algorithm specific')


class CompositionalLearner(Learner):
    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        self.save_data(0, task_id, testloaders)
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs,
                            save_freq, testloaders)
        else:
            self.net.freeze_modules()
            self.net.freeze_structure()     # freeze structure for all tasks
            self.net.freeze_structure(
                freeze=False, task_id=task_id)    # except current one
            iter_cnt = 0
            for i in range(num_epochs):
                if (i + 1) % component_update_freq == 0:
                    # replace one structure epoch with one module epoch
                    self.update_modules(trainloader, task_id)
                else:
                    for X, Y in trainloader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        self.update_structure(X, Y, task_id)
                        iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    self.save_data(i + 1, task_id, testloaders)

            self.save_data(num_epochs + 1, task_id,
                           testloaders, final_save=True)
            self.update_multitask_cost(trainloader, task_id)

    def update_structure(self, X, Y, task_id):
        # assume shared parameters are frozen and just take a gradient step on the structure
        self.gradient_step(X, Y, task_id)

    def update_modules(self, *args, **kwargs):
        raise NotImplementedError('Update modules is algorithm specific')


class CompositionalDynamicLearner(CompositionalLearner):
    def train(self, trainloader, task_id, valloader, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        eval_bool = testloaders is not None
        self.save_data(0, task_id, testloaders)
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs,
                            save_freq, testloaders)
        else:
            self.net.freeze_modules()
            self.net.freeze_structure()     # freeze structure for all tasks
            # freeze original modules and structure
            self.net.add_tmp_module(task_id)

            self.optimizer.add_param_group(
                {'params': self.net.components[-1].parameters()})

            if hasattr(self, 'preconditioner'):
                self.preconditioner.add_module(self.net.components[-1])

            # unfreeze (new) structure for current task
            self.net.freeze_structure(freeze=False, task_id=task_id)
            iter_cnt = 0

            for i in range(num_epochs):
                if (i + 1) % component_update_freq == 0:
                    self.update_modules(trainloader, task_id)
                else:
                    for X, Y in trainloader:
                        X_cpu, Y_cpu = X, Y
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        self.update_structure(X, Y, task_id)
                        self.net.hide_tmp_module()
                        self.update_structure(X, Y, task_id)
                        self.net.recover_hidden_module()
                        iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    self.save_data(i + 1, task_id, testloaders)
            self.conditionally_add_module(valloader, task_id)
            self.save_data(num_epochs + 1, task_id,
                           testloaders, final_save=True)
            self.update_multitask_cost(trainloader, task_id)

    def conditionally_add_module(self, valloader, task_id):
        test_loss = self.test_loss
        test_acc = self.test_acc

        self.evaluate({task_id: valloader})
        update_loss, no_update_loss = self.test_loss[task_id]
        update_acc, no_update_acc = self.test_acc[task_id]
        logging.info(
            'W/update: {}, WO/update: {}'.format(update_acc, no_update_acc))
        if no_update_acc == 0 or (update_acc - no_update_acc) / no_update_acc > self.improvement_threshold:
            logging.info('Keeping new module. Total: {}'.format(
                self.net.num_components))
        else:
            self.net.remove_tmp_module()
            logging.info('Not keeping new module. Total: {}'.format(
                self.net.num_components))

        self.test_loss = test_loss
        self.test_acc = test_acc

    def evaluate(self, testloaders, eval_no_update=True):
        was_training = self.net.training
        prev_reduction = self.loss.reduction
        self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        self.net.eval()
        with torch.no_grad():
            self.test_loss = {}
            self.test_acc = {}
            for task, loader in testloaders.items():
                l = 0.
                a = 0.
                n = len(loader.dataset)
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task)
                    l += self.loss(Y_hat, Y).item()
                    a += (Y_hat.argmax(dim=1) == Y).sum().item()
                if eval_no_update and task == self.T - 1 and self.T > self.net.num_init_tasks:
                    self.net.hide_tmp_module()
                    l1 = 0.
                    a1 = 0.
                    for X, Y in loader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        Y_hat = self.net(X, task)
                        l1 += self.loss(Y_hat, Y).item()
                        a1 += (Y_hat.argmax(dim=1) == Y).sum().item()
                    self.test_loss[task] = (l / n, l1 / n)
                    self.test_acc[task] = (a / n, a1 / n)
                    self.net.recover_hidden_module()
                else:
                    self.test_loss[task] = l / n
                    self.test_acc[task] = a / n

        self.loss.reduction = prev_reduction
        if was_training:
            self.net.train()

    def save_data(self, epoch, task_id, testloaders,  final_save=False):
        super().save_data(epoch, task_id, testloaders, final_save=final_save)
        if final_save:
            logging.info('final components: {}'.format(
                self.net.num_components))
