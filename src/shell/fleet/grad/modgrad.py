'''
File: /modgrad.py
Project: shell
Created Date: Monday March 27th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import logging
from copy import deepcopy
import ray
from shell.fleet.grad.monograd import ModelSyncAgent, ParallelModelSyncAgent
import copy
import torch
from torch.utils.data.dataset import ConcatDataset
from shell.datasets.datasets import get_custom_tensordataset


class ModGrad(ModelSyncAgent):
    def prepare_model(self):
        # NOTE: we only share the initial components, so we remove
        # anything more than that
        num_init_components = self.net.depth
        # add to self.excluded_params
        num_components = len(self.net.components)
        for i in range(num_init_components, num_components):
            self.excluded_params.add("components.{}".format(i))
        return super().prepare_model()

    def process_communicate(self, task_id, communication_round, final=False):
        # TODO: maybe only re-optimize past structures only if 

        # if communication_round % self.sharing_strategy.log_freq == 0:
        #     self.log(task_id, communication_round)

        print(f"node {self.node_id} is processing at round {communication_round} for task {task_id}")
        # TODO: this logging is potentiallly buggy
        # self.log(task_id, communication_round)
        # self.aggregate_models()
        # # self.log(task_id, communication_round+1)
        # self.log(task_id, communication_round)

        self.log(task_id, communication_round, info={'info': 'before'})
        self.aggregate_models()
        self.log(task_id, communication_round, info={'info': 'after'})


        # # ModGrad: retrain on local tasks using experience replay. Update ONLY shared modules,
        # # keeping structures and other task-specific modules fixed.
        # trainloader = (
        #     torch.utils.data.DataLoader(self.dataset.trainset[task_id],
        #                                 batch_size=self.batch_size,
        #                                 shuffle=True,
        #                                 num_workers=4,
        #                                 pin_memory=True,
        #                                 ))

        # task_id_retrain = f"{task_id}_retrain_round_{communication_round}"
        # testloaders = {task: torch.utils.data.DataLoader(testset,
        #                                                  batch_size=128,
        #                                                  shuffle=False,
        #                                                  num_workers=0,
        #                                                  pin_memory=True,
        #                                                  ) for task, testset in enumerate(self.dataset.testset[:(task_id+1)])}
        # self.finetune_shared_modules(
        #     trainloader, task_id, testloaders, task_id_retrain=task_id_retrain)

        if self.sharing_strategy.when_reoptimize_structure == 'never':
            return
        elif self.sharing_strategy.when_reoptimize_structure == 'final' and final:
            self.reoptimize_past_structures(task_id, communication_round)
        elif self.sharing_strategy.when_reoptimize_structure == 'always':
            # always re-optimize
            self.reoptimize_past_structures(task_id, communication_round)
        else:
            raise NotImplementedError(f'when_reoptimize_structure: {self.sharing_strategy.when_reoptimize_structure} not implemented')

        # self.agent.save_data(self.sharing_strategy.retrain.num_epochs + 1, task_id_retrain,
        #                      testloaders, final_save=False)  # final eval

    def reoptimize_past_structures(self, task_id, communication_round, num_epochs=1):
        # optimize structures from 0--> task_id - 1 (excluding current task_id)
        # using the replay buffer
        # print("REOPTIMIZE TASK_ID:", task_id, len(self.agent.memory_loaders))
        assert len(self.agent.memory_loaders.values()) == task_id, f"task_id: {task_id}, len(self.agent.memory_loaders.values()): {len(self.agent.memory_loaders.values())}"

        # TODO: handle the component dropout carefully. 
        # print("BEFORE CURRENT_ACTIVE_CANDIDATE_INDEX", self.net.active_candidate_index)
        current_active_candidate_index = self.net.active_candidate_index
        self.net.active_candidate_index = None

        # NOTE: at the final communication round for the task, we need to reset the adam optimizer.
        # Adam uses a moving average of the past gradients, so if the model dynamically changes (e.g.,
        # we remove the conditional component), we will get an error.
        # This wasn't a problem before for some reasons, although re-optimizing past modules (using experience
        # replay) seems like it should cause the same problem.
        # if communication_round == self.num_coms[task_id]-1:
        if task_id not in self.num_coms: 
        # HACK: hacky way to identify the final communication round basically
        # for the task. The task_id will be +1 the actual task because
        # we've completed the last training epoch (communication happens AFTER
        # training). So the current task is "just" considered past.
            # print("RESETING OPTIMIZER")
            self.agent.optimizer = torch.optim.Adam(self.net.parameters(),)
        # print("STRUCTURE", self.net.structure)
        for _ in range(num_epochs):
            for task in range(task_id):
                self.net.freeze_modules()
                self.net.freeze_structure()
                self.net.unfreeze_structure(task_id=task)

                # update structure for task
                loader = torch.utils.data.DataLoader(get_custom_tensordataset(self.agent.memory_loaders[task].dataset.tensors,
                                                  name=self.agent.dataset_name,
                                      use_contrastive=self.agent.use_contrastive),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True,
                )

                for X, Y, _ in loader:
                    if isinstance(X, list):
                        # contrastive two views
                        X = torch.cat([X[0], X[1]], dim=0)
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    # with new module
                    # print(self.net.structure)
                    self.agent.update_structure(
                        X, Y, task)
        self.net.active_candidate_index = current_active_candidate_index
        # print("AFTER CURRENT_ACTIVE_CANDIDATE_INDEX", self.net.active_candidate_index)
         
    def finetune_shared_modules(self, trainloader, task_id, testloaders, train_mode=None,
                                task_id_retrain="", save_freq=1):
        """
        Retrain:
        - the shared modules.
        - all task-specific decoders

        Fixing:
        - structures


        NOTE: only tested for MNIST VARIANTS
        """
        self.net.freeze_structure()

        # freeze all the modules except the shared ones and the task decoder.
        self.net.freeze_modules()
        self.net.unfreeze_some_modules(range(self.net.depth))
        # self.net.unfreeze_decoder(task_id)

        for t in range(task_id+1):
            self.net.unfreeze_decoder(t)

        # training
        prev_reduction = self.agent.get_loss_reduction()
        self.agent.set_loss_reduction('sum')

        tmp_dataset = copy.copy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + \
            (torch.full((len(tmp_dataset),), task_id, dtype=int),)

        mega_dataset = ConcatDataset(
            [get_custom_tensordataset(loader.dataset.tensors, name=self.agent.dataset_name,
                                      use_contrastive=self.agent.use_contrastive) for loader in self.agent.memory_loaders.values()])
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
                                                  batch_size=self.agent.memory_loaders[0].batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True
                                                  )

        for epoch in range(self.sharing_strategy.retrain.num_epochs):
            for X, Y, t in mega_loader:
                if isinstance(X, list):
                    # contrastive two views
                    X = torch.cat([X[0], X[1]], dim=0)
                X = X.to(self.net.device, non_blocking=True)
                Y = Y.to(self.net.device, non_blocking=True)
                l = 0.
                n = 0
                all_t = torch.unique(t)

                if self.agent.use_contrastive:
                    Xhaf = X[:len(X)//2]
                    Xother = X[len(X)//2:]

                for task_id_tmp in all_t:
                    Yt = Y[t == task_id_tmp]
                    if self.agent.use_contrastive:
                        # Xt will be twice as long as Yt
                        # use advanced indexing to get the first half
                        Xt_haf = Xhaf[t == task_id_tmp]
                        Xt_other = Xother[t == task_id_tmp]
                        Xt = torch.cat([Xt_haf, Xt_other], dim=0)
                    else:
                        Xt = X[t == task_id_tmp]
                    l += self.agent.compute_loss(Xt,
                                                 Yt, task_id_tmp,
                                                 mode=train_mode)
                    n += X.shape[0]
                l /= n
                self.agent.optimizer.zero_grad()
                l.backward()
                self.agent.optimizer.step()
            if epoch % save_freq == 0:
                self.agent.save_data(epoch + 1, task_id_retrain, testloaders)

        # undo all the freezing stuff
        self.agent.set_loss_reduction(prev_reduction)
        self.net.freeze_decoder()
        self.net.freeze_modules()
        self.net.unfreeze_structure(task_id)


@ray.remote
class ParallelModGrad(ModGrad):
    def communicate(self, task_id, communication_round, final=False):
        # logging.info(
        #     f"node {self.node_id} is communicating at round {communication_round} for task {task_id}")
        # TODO: Should we do deepcopy???
        # put model on object store
        # state_dict = deepcopy(self.net.state_dict())
        # model = state_dict
        # model = ray.put(state_dict)
        # send model to neighbors
        # logging.info(f"My neighbors are: {self.neighbors}")
        for neighbor in self.neighbors:
            # neighbor_id = ray.get(neighbor.get_node_id.remote())
            # NOTE: neighbor_id for some reason is NOT responding...
            # logging.info(f"SENDING MODEL: {self.node_id} -> {neighbor_id}")
            ray.get(neighbor.receive.remote(self.node_id, self.model, "model"))
            self.bytes_sent[(task_id, communication_round)
                            ] = self.compute_model_size(self.model)
