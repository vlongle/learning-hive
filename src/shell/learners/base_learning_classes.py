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
from shell.utils.supcontrast import SupConLoss
from shell.utils.oodloss import OODSeparationLoss
from shell.learners.fl_utils import *
from torch.utils.data.dataset import ConcatDataset
from shell.datasets.datasets import CustomConcatTensorDataset
import copy
# write custom RandomGrayScale that operate on a per image
# basis


class Learner():
    def __init__(self, net, save_dir='./tmp/results/',
                 improvement_threshold=0.05,
                 use_contrastive=False, dataset_name=None, fl_strategy=None,
                 mu=None, use_ood_separation_loss=False, lambda_ood=2.0,
                 delta_ood=1.0):
        self.net = net
        self.shared_replay_buffers = {}  # received from neighbors
        self.ce_loss = nn.CrossEntropyLoss()
        self.use_contrastive = use_contrastive
        self.dataset_name = dataset_name
        if use_contrastive:
            # temperature = 0.1 if dataset_name == 'cifar100' else 0.06
            # temperature = 0.01 if dataset_name == 'cifar100' else 0.06
            # temperature = 0.07 if dataset_name == 'cifar100' else 0.06
            temperature = 0.06
            self.sup_loss = SupConLoss(temperature=temperature)

        # self.loss = nn.BCEWithLogitsLoss() if net.binary else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        print('optimizer', self.optimizer)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        #   lr=0.005)
        self.improvement_threshold = improvement_threshold
        self.T = 0
        self.observed_tasks = set()

        self.save_dir = create_dir_if_not_exist(save_dir)
        self.record = Record(os.path.join(self.save_dir, "record.csv"))
        self.dynamic_record = Record(os.path.join(
            self.save_dir, "add_modules_record.csv"))
        self.sharing_data_record = Record(os.path.join(
            self.save_dir, "sharing_data_record.csv"))

        self.writer = SummaryWriter(
            log_dir=create_dir_if_not_exist(os.path.join(self.save_dir, "tensorboard/")))
        self.init_trainloaders = None

        self.mode = "ce"
        if self.use_contrastive:
            self.mode = "both"
        self.fl_strategy = fl_strategy
        self.mu = self.global_model = None

        if fl_strategy is not None:
            self.global_model = None
            self.mu = mu

        self.use_ood_separation_loss = use_ood_separation_loss
        self.lambda_ood = lambda_ood
        self.delta_ood = delta_ood
        self.ood_loss = OODSeparationLoss(lambda_ood=self.lambda_ood,
                                          delta=self.delta_ood)
        self.ood_data = {}

    def change_save_dir(self, save_dir):
        self.save_dir = create_dir_if_not_exist(save_dir)
        self.record = Record(os.path.join(self.save_dir, "record.csv"))
        self.dynamic_record = Record(os.path.join(
            self.save_dir, "add_modules_record.csv"))
        self.sharing_data_record = Record(os.path.join(
            self.save_dir, "sharing_data_record.csv"))
        self.writer = SummaryWriter(
            log_dir=create_dir_if_not_exist(os.path.join(self.save_dir, "tensorboard/")))

    # def apply_transform(self, X):
    #     # X: (batch_size, n_channels, height, width)
    #     # apply self.transform to each image in X
    #     # return: (batch_size, n_channels, height, width)
    #     return self.train_transform(X)
        # morally correct way but it's too slow
        # return torch.stack([self.train_transform(x) for x in X])

    def record_shared_data_stats(self, train_task_id, epoch):
        for task_id, replay in sorted(self.shared_replay_buffers.items()):
            self.sharing_data_record.write(
                {
                    'train_task': train_task_id,
                    'task_id': task_id,
                    'epoch': epoch,
                    'num_samples': len(replay),
                }
            )

        self.sharing_data_record.save()

    # def make_shared_memory_loaders(self, batch_size=32):
    #     self.shared_memory_loaders = {}
    #     for task_id in self.shared_replay_buffers.keys():
    #         if len(self.shared_replay_buffers[task_id]) == 0:
    #             continue
    #         self.shared_memory_loaders[task_id] = (
    #             torch.utils.data.DataLoader(self.shared_replay_buffers[task_id],
    #                                         batch_size=batch_size,
    #                                         shuffle=True,
    #                                         num_workers=1,
    #                                         pin_memory=True
    #                                         ))

    def get_loss_reduction(self):
        if self.use_contrastive:
            assert self.ce_loss.reduction == self.sup_loss.reduction
        return self.ce_loss.reduction

    def set_loss_reduction(self, reduction):
        self.ce_loss.reduction = reduction
        if self.use_contrastive:
            self.sup_loss.reduction = reduction

    def compute_contrastive_loss(self, X, Y, task_id):
        assert len(X) == 2 * len(Y)
        encoded_X = self.net.contrastive_embedding(X, task_id)
        # encoded_transformed_X = self.net.contrastive_embedding(
        #     self.apply_transform(X), task_id)  # (N_samples, N_features)
        # features.shape = (N_samples, N_views=2, N_features)
        encoded_X, encoded_transformed_X = torch.split(
            encoded_X, encoded_X.shape[0] // 2, dim=0)
        features = torch.cat(
            [encoded_X.unsqueeze(1), encoded_transformed_X.unsqueeze(1)], dim=1)
        cl = self.sup_loss(features, labels=Y)
        # if cl is nan, then exit
        if torch.isnan(cl):
            logging.error("Contrastive loss is nan")
            exit(1)
        return cl

    def compute_cross_entropy_loss(self, X, Y, task_id, detach=True):
        # =============================
        # Cross entropy loss
        # NOTE: detach so that cross_entropy does not propagate gradients back to the representation learner
        if X.shape[0] != Y.shape[0]:
            # using contrastive so we have two views
            X, _ = torch.split(X, X.shape[0] // 2, dim=0)
        X_encode = self.net.encode(X, task_id)
        if detach:
            X_encode = X_encode.detach()
        Y_hat = self.net.decoder[task_id](X_encode)
        # print("structure", self.net.structure[task_id])
        # print('decoder', self.net.decoder[task_id].bias)
        # print('comp[0]', self.net.components[0].bias)
        # print('Y_hat', Y_hat)
        # check if Y is float if yes, raise error
        # if Y.dtype == torch.float32:
        #     print('Y:', Y)
        #     raise ValueError(
        #         "?????????????/// Y is float32, make sure to convert to long before passing to compute_cross_entropy_loss")
        # # check that Y is either 0 or 1
        # if Y.max() > 1 or Y.min() < 0:
        #     print('Y:', Y)
        #     raise ValueError(
        #         "?????????????/// Y is not binary, make sure to convert to binary before passing to compute_cross_entropy_loss")

        ce = self.ce_loss(Y_hat, Y)
        return ce

    # def compute_auxillary_loss(self, X, Y, task_id):
    #     loss = 0.
    #     if self.fl_strategy is not None:
    #         if self.fl_strategy == "fedprox":
    #             loss += compute_fedprox_aux_loss(local_model=self.net, global_model=self.global_model,
    #                                              mu=self.mu)
    #         else:
    #             raise NotImplementedError(
    #                 "FL strategy %s not implemented" % self.fl_strategy)

    #     if self.use_ood_separation_loss:
    #         if X.shape[0] != Y.shape[0]:
    #             # using contrastive so we have two views
    #             X, _ = torch.split(X, X.shape[0] // 2, dim=0)
    #         X_encode = self.net.encode(X, task_id)
    #         # task_id might be a one value tensor, convert to int
    #         if isinstance(task_id, torch.Tensor):
    #             task_id = task_id.item()
    #         X_ood, *_ = self.ood_data[task_id]
    #         X_ood = X_ood.to(self.net.device, non_blocking=True)
    #         ood_encode = self.net.encode(X_ood, task_id)
    #         loss += self.ood_loss(X_encode, ood_encode)

        # return loss

    def compute_loss(self, X, Y, task_id, mode=None, log=False, global_step=None, use_aux=False):
        """
        Compute main loss + (optional aux loss for FL)
        """
        loss = self.compute_task_loss(X, Y, task_id, mode=mode, log=log)
        # self.writer.add_scalar(
        #     f'loss/train_{self.T-1}/task_{task_id}/loss', loss, global_step)
        # logging.info("before %s", loss)
        # print("task_loss:", loss, "mu", self.mu)
        # if use_aux:
        #     aux_loss = self.compute_auxillary_loss(X, Y, task_id)
        #     self.writer.add_scalar(
        #         f'aux_loss/train_{self.T-1}/task_{task_id}/loss', aux_loss, global_step)
        #     loss += aux_loss

        # print("combined loss:", loss)

        # save loss to self.log_file
        # NOTE: DEBUG
        # logging.info(loss.item())
        return loss

    def compute_task_loss(self, X, Y, task_id, mode=None, log=False):
        """
        Compute cross_entropy + supcon loss. Make sure that 
        cross_entropy does not propagate gradients back
        to the representation learner.
        """
        if mode is None:
            mode = self.mode

        if mode == "both":
            # jointly train ce and supcon loss
            ce = self.compute_cross_entropy_loss(X, Y, task_id, detach=True)
            cl = self.compute_contrastive_loss(X, Y, task_id)
            scale = 1.0
            loss = ce + scale * cl
            # logging.info("ce %s cl %s l %s", ce, cl, loss)
            return loss
        elif mode == "ce":
            # only train ce (backpropage through the entire model)
            return self.compute_cross_entropy_loss(X, Y, task_id, detach=False)
        elif mode == "finetune_ce":
            # train ce to only finetune the last layer (stop gradient so the rest of the model is not updated)
            return self.compute_cross_entropy_loss(X, Y, task_id, detach=True)
        elif mode == "cl":
            # only train supcon loss
            return self.compute_contrastive_loss(X, Y, task_id)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

    def train(self, *args, **kwargs):
        raise NotImplementedError('Training loop is algorithm specific')

    def init_train(self, trainloader, task_id, start_epoch, num_epochs, save_freq=1, testloaders=None,
                   final=True):
        if self.init_trainloaders is None:
            self.init_trainloaders = {}
        self.init_trainloaders[task_id] = trainloader
        if len(self.init_trainloaders) == self.net.num_init_tasks:
            # print([(len(loader), len(loader.dataset))
            #       for loader in self.init_trainloaders.values()])
            for i in range(start_epoch, num_epochs + start_epoch):
                # print('epoch', i)
                # for j in range(len(self.net.components)):
                #     print('\t component', j)
                #     print(
                #         '\t', self.net.components[j].bias.mean(), self.net.components[j].weight.mean())
                #     print('\t', self.net.structure[j])
                #     print('\t', self.net.decoder[j].bias.mean(),
                #           self.net.decoder[j].weight.mean())
                for XY_all in zip_longest(*self.init_trainloaders.values()):
                    for task, XY in zip(self.init_trainloaders.keys(), XY_all):
                        if XY is not None:
                            X, Y = XY
                            if isinstance(X, list):
                                # contrastive two views
                                X = torch.cat([X[0], X[1]], dim=0)
                            # print('BEFORE: comp[0]',
                            #       self.net.components[0].bias)
                            X = X.to(self.net.device, non_blocking=True)
                            Y = Y.to(self.net.device, non_blocking=True)
                            self.gradient_step(X, Y, task, global_step=i)
                            # print('AFTER: comp[0]',
                            #       self.net.components[0].bias)
                # exit(0)
                if i % save_freq == 0:
                    self.save_data(i + 1, task_id, testloaders)

            if final:
                self.save_data(num_epochs + start_epoch + 1, task_id,
                               testloaders, final_save=final)
                for task, loader in self.init_trainloaders.items():
                    self.update_multitask_cost(loader, task)
        else:
            self.save_data(start_epoch, task_id,
                           testloaders, final_save=final)
        # print('DONE init train task', task_id, 'rand torch seed', int(torch.empty(
        #     (), dtype=torch.int64).random_().item()))
        # self.update_multitask_cost(self.init_trainloaders[task_id], task_id)

    def evaluate(self, testloaders, mode=None, eval_no_update=True):
        was_training = self.net.training
        prev_reduction = self.get_loss_reduction()
        # self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        # make sure the loss is summed over instances
        self.set_loss_reduction('sum')
        self.net.eval()
        with torch.no_grad():
            test_loss = {}
            test_acc = {}
            for task, loader in testloaders.items():
                l = 0.
                a = 0.
                n = len(loader.dataset)
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task)
                    l += self.compute_task_loss(X, Y, task, mode='ce').item()
                    a += (Y_hat.argmax(dim=1) == Y).sum().item()
                    # a += ((Y_hat > 0) == (Y == 1)
                    #       if self.net.binary else Y_hat.argmax(dim=1) == Y).sum().item()

                test_loss[task] = l / n
                test_acc[task] = a / n

        self.set_loss_reduction(prev_reduction)
        if was_training:
            self.net.train()
        return test_loss, test_acc

    def gradient_step(self, X, Y, task_id, train_mode=None, global_step=None):
        # Y_hat = self.net(X, task_id=task_id)
        X = X.to(self.net.device, non_blocking=True)
        Y = Y.to(self.net.device, non_blocking=True)
        # print("task_id:", task_id, 'y', Y[:5])
        l = self.compute_loss(X, Y, task_id, mode=train_mode,
                              log=True, global_step=global_step)
        # exit(0)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        # print(task_id, "LOSS:", l, 'exp_avg', list(
        #     self.optimizer.state.values())[0]['exp_avg'].mean())

        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(
        #             f"Gradient for {name}: {param.grad.mean() if param.grad is not None else 'No grad'}")

        # exit(0)

    def save_data(self, epoch, task_id, testloaders, final_save=False, mode=None,
                  save_dir=None, record=None):
        if record is None:
            record = self.record
        if save_dir is None:
            save_dir = self.save_dir
        self.test_loss, self.test_acc = self.evaluate(testloaders, mode=mode)
        task_results_dir = os.path.join(
            save_dir, 'task_{}'.format(task_id))
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
            line = '\ttask: {}\tloss: {:.8f}\tacc: {:.3f}'.format(
                task, self.test_loss[task], self.test_acc[task])
            logging.info(line)
            record.write(
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
            record.save()

        self.record_shared_data_stats(task_id, epoch)

    def update_multitask_cost(self, loader, task_id):
        raise NotImplementedError(
            'Update update_multitask is algorithm specific')


class CompositionalLearner(Learner):
    def update_structure(self, X, Y, task_id, train_mode=None, global_step=None):
        # assume shared parameters are frozen and just take a gradient step on the structure
        self.gradient_step(
            X, Y, task_id, train_mode=train_mode, global_step=global_step)

    def update_modules(self, *args, **kwargs):
        raise NotImplementedError('Update modules is algorithm specific')


class CompositionalDynamicLearner(CompositionalLearner):
    def train(self, trainloader, task_id, valloader,
              component_update_freq=100, start_epoch=0, num_epochs=100, save_freq=1, testloaders=None,
              train_mode=None, num_candidate_modules=None, module_list=None,
              final=True,
              train_candidate_module=True,):
        # logging.info('task_id %s len(self.net.components) %s', task_id, len(self.net.components))
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        if start_epoch == 0:
            # zeroshot
            self.save_data(start_epoch, task_id, testloaders, mode=train_mode)

        if self.T <= self.net.num_init_tasks:
            # NOTE: doesn't need to freeze_structure because one_hot structure
            # will be fixed anyway. We need the decoder to change for
            # the new contrastive learning paradigm.
            # self.net.freeze_structure()
            # NOTE: we're keeping the decoder unfrozen!
            # and freeze structure just in case we're using one-hot same structure
            # for all the tasks!
            self.net.freeze_linear_weights()
            # self.net.freeze_structure()
            self.init_train(trainloader, task_id, start_epoch, num_epochs,
                            save_freq, testloaders, final)
        else:
            self.net.freeze_modules()
            self.net.freeze_structure()     # freeze structure for all tasks
            # freeze original modules and structure

            # self.net.add_tmp_module(task_id)
            # self.optimizer.add_param_group(
            #     {'params': self.net.components[-1].parameters()})

            if start_epoch == 0:
                if module_list is None:
                    module_list = []
                if num_candidate_modules is None:
                    num_candidate_modules = len(module_list) + 1

                print("no. current components", len(self.net.components),
                      "NUM_CANDIDATE_MODULES", num_candidate_modules,
                      'len(module_list)', len(module_list))
                # print('rand torch seed', int(torch.empty(
                #     (), dtype=torch.int64).random_().item()))
                self.net.add_tmp_modules(task_id, num_candidate_modules)
                self.net.receive_modules(task_id, module_list)

                # self.optimizer = torch.optim.Adam(self.net.parameters(),)
                # the last num_candidate_modules components
                # for idx in range(-num_candidate_modules, 0, 1):

                for idx in self.net.candidate_indices:
                    # print('adding param group', idx)
                    self.optimizer.add_param_group(
                        {'params': self.net.components[idx].parameters()})

                # print('new comps:', self.net.components[-1].bias)
                # print('structure', self.net.structure[task_id])
                # exit(0)

            self.net.unfreeze_structure(task_id=task_id)

            if task_id in self.shared_replay_buffers:
                tmp_dataset = copy.deepcopy(trainloader.dataset)
                X, y, _ = self.shared_replay_buffers[task_id].get_tensors()
                mega_dataset = CustomConcatTensorDataset(
                    (X, y), tmp_dataset.tensors)
                trainloader = torch.utils.data.DataLoader(mega_dataset,
                                                          batch_size=trainloader.batch_size,
                                                          shuffle=True,
                                                          num_workers=2,
                                                          pin_memory=True
                                                          )

            for i in range(start_epoch, num_epochs + start_epoch):
                # print('num_epochs', num_epochs, 'start_epoch', start_epoch, 'i', i)
                if (i + 1) % component_update_freq == 0:
                    # print('UPDATING MODULES')
                    self.update_modules(
                        trainloader, task_id, train_mode=train_mode, global_step=i,)

                else:
                    for X, Y in trainloader:
                        if isinstance(X, list):
                            # contrastive two views
                            X = torch.cat([X[0], X[1]], dim=0)
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)

                        # with new module. Update struct + update the active
                        # candidate
                        # if train_candidate_module:
                        #     self.net.unfreeze_module(
                        #         self.net.active_candidate_index)
                        self.update_structure(
                            X, Y, task_id, train_mode=train_mode,
                            global_step=i)
                        # self.net.hide_tmp_module()

                        # without new module
                        # self.net.freeze_module(self.net.active_candidate_index)
                        self.net.hide_tmp_modulev2()
                        self.update_structure(
                            X, Y, task_id, train_mode=train_mode,
                            global_step=i)
                        # self.net.recover_hidden_module()
                        self.net.recover_hidden_modulev2()
                        self.net.select_active_module()  # select the next module in round-robin
                if i % save_freq == 0:
                    self.save_data(i + 1, task_id, testloaders,
                                   mode=train_mode)
            if final:
                self.conditionally_add_module(valloader, task_id)
                self.save_data(num_epochs + start_epoch + 1, task_id,
                               testloaders, final_save=final, mode=train_mode)
                self.update_multitask_cost(trainloader, task_id)

    def conditionally_add_module(self, valloader, task_id):
        performances = {}  # relative improvement for each candidate
        losses = {}

        # Set the active index to the first candidate module
        # reset active module to the first one
        self.net.select_active_module(self.net.candidate_indices[0])

        for idx in self.net.candidate_indices:
            self.test_loss, self.test_acc = self.evaluate({task_id: valloader})
            update_acc, no_update_acc = self.test_acc[task_id]
            performances[idx] = (update_acc - no_update_acc) / no_update_acc
            logging.info(
                'candidate {}: W/update: {}, WO/update: {}, improv {}'.format(idx, update_acc, no_update_acc,
                                                                              performances[idx]))
            self.net.select_active_module()  # select the next module in round-robin

            losses[idx] = self.test_loss[task_id]

        # Decide the best candidate based on relative improvement
        best_candidate_idx = max(performances, key=performances.get)
        best_improvement = performances[best_candidate_idx]

        num_candidate_modules = len(self.net.candidate_indices)

        # Check if the improvement is greater than the threshold, and if not, remove all candidates
        if best_improvement <= self.improvement_threshold:
            self.net.remove_tmp_modulev2(self.net.candidate_indices)
            logging.info('Not keeping any new modules. Total: {}'.format(
                self.net.num_components))
            add_new_module = False
        else:
            # Keep the best candidate and remove others
            exclude_indices = [
                idx for idx in self.net.candidate_indices if idx != best_candidate_idx]
            self.net.remove_tmp_modulev2(exclude_indices)
            logging.info('Keeping new module {}. Total: {}'.format(best_candidate_idx,
                                                                   self.net.num_components))
            add_new_module = True

        self.dynamic_record.write(
            {
                'task_id': task_id,
                'best_candidate_idx': best_candidate_idx,
                'num_candidates': num_candidate_modules,
                'best_improvement': best_improvement,
                'num_components': self.net.num_components,
                'add_new_module': add_new_module,
            }
        )

        self.dynamic_record.save()

        return performances, losses

    def evaluate(self, testloaders, eval_no_update=True, mode=None):
        was_training = self.net.training
        # prev_reduction = self.loss.reduction
        prev_reduction = self.get_loss_reduction()
        # self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        self.set_loss_reduction('sum')
        self.net.eval()
        with torch.no_grad():
            test_loss = {}
            test_acc = {}
            for task, loader in testloaders.items():
                l = 0.
                a = 0.
                n = len(loader.dataset)
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task)
                    l += self.compute_task_loss(X, Y, task, mode='ce').item()
                    a += (Y_hat.argmax(dim=1) == Y).sum().item()
                    # a += ((Y_hat > 0) == (Y == 1)
                    #       if self.net.binary else Y_hat.argmax(dim=1) == Y).sum().item()
                # NOTE: only go to this loop for task above the num_init_tasks
                if (eval_no_update and task == self.T - 1 and self.T > self.net.num_init_tasks) and self.net.last_active_candidate_index:
                    # self.net.hide_tmp_module()
                    self.net.hide_tmp_modulev2()
                    l1 = 0.
                    a1 = 0.
                    for X, Y in loader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        Y_hat = self.net(X, task)
                        l1 += self.compute_task_loss(X,
                                                     Y, task, mode='ce').item()
                        a1 += (Y_hat.argmax(dim=1) == Y).sum().item()
                        # a1 += ((Y_hat > 0) == (Y == 1)
                        #        if self.net.binary else Y_hat.argmax(dim=1) == Y).sum().item()
                    test_loss[task] = (l / n, l1 / n)
                    test_acc[task] = (a / n, a1 / n)
                    # self.net.recover_hidden_module()
                    self.net.recover_hidden_modulev2()
                    # print(
                    #     f"dropout test_acc {self.test_acc[task]}, structure {self.net.structure[task]}")
                    # print(
                    #     f"new components {self.net.components[-1].weight[:5]}")
                else:
                    test_loss[task] = l / n
                    test_acc[task] = a / n

        # self.loss.reduction = prev_reduction
        self.set_loss_reduction(prev_reduction)
        if was_training:
            self.net.train()
        return test_loss, test_acc

    def save_data(self, epoch, task_id, testloaders,  final_save=False, mode=None, save_dir=None,
                  record=None):
        super().save_data(epoch, task_id, testloaders,
                          final_save=final_save, mode=mode, save_dir=save_dir, record=record)
        if final_save:
            logging.info('final components: {}'.format(
                self.net.num_components))
