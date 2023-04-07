'''
File: /base_net_classes.py
Project: models
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import torch
import torch.nn as nn
import numpy as np


class CompositionalNet(nn.Module):
    def __init__(self,
                 i_size,
                 depth,
                 num_classes,
                 num_tasks,
                 num_init_tasks=None,
                 init_ordering_mode='random_onehot',
                 device='cuda'):
        super().__init__()
        self.device = device
        self.depth = depth
        self.num_tasks = num_tasks
        if num_init_tasks is None:
            num_init_tasks = depth
        self.num_init_tasks = num_init_tasks
        self.init_ordering_mode = init_ordering_mode
        self.i_size = i_size
        if isinstance(self.i_size, int):
            self.i_size = [self.i_size] * num_tasks
        self.num_classes = num_classes
        if isinstance(self.num_classes, int):
            self.num_classes = [self.num_classes] * num_tasks

    def init_ordering(self):
        raise NotImplementedError(
            'Init ordering must be architecture specific')

    '''
    In both freeze functions we need to take care of making 
    grad=None for any parameter that does not require_grad.
    The zero_grad() function does not take care of this, and
    otherwise Adam will treat non-updates as updates (because
    grad is not None)
    '''

    def freeze_modules(self):
        for param in self.components.parameters():
            param.requires_grad = False
            param.grad = None
        # if hasattr(self, 'projector'):
        #     self.freeze_projector()

    def freeze_projector(self):
        for param in self.projector.parameters():
            param.requires_grad = False
            param.grad = None

    def unfreeze_modules(self):
        for param in self.components.parameters():
            param.requires_grad = True
        # if hasattr(self, 'projector'):
        #     self.unfreeze_projector()

    def unfreeze_projector(self):
        for param in self.projector.parameters():
            param.requires_grad = True

    def unfreeze_some_modules(self, list_of_modules):
        for i in list_of_modules:
            for param in self.components[i].parameters():
                param.requires_grad = True

    def freeze_structure(self, freeze=True):
        raise NotImplementedError(
            'Freeze structure must be architecture specific')


class SoftOrderingNet(CompositionalNet):
    def __init__(self,
                 i_size,
                 depth,
                 num_classes,
                 num_tasks,
                 num_init_tasks=None,
                 init_ordering_mode='random_onehot',
                 device='cuda'):
        super().__init__(i_size,
                         depth,
                         num_classes,
                         num_tasks,
                         num_init_tasks=num_init_tasks,
                         init_ordering_mode=init_ordering_mode,
                         device=device)

        self.structure = nn.ParameterList([nn.Parameter(torch.ones(
            self.depth, self.depth)) for t in range(self.num_tasks)])
        self.init_ordering()

        self.softmax = nn.Softmax(dim=0)

    def init_ordering(self):
        if self.init_ordering_mode == 'one_module_per_task':
            assert self.num_init_tasks == self.depth, \
                'Initializing one module per task requires the number of initialization tasks to be the same as the depth'
            # first "k" tasks, use sinle layer repeated
            for t in range(self.num_init_tasks):
                self.structure[t].data = -np.inf * \
                    torch.ones(self.depth, self.depth)
                self.structure[t].data[t, :] = 1
        elif self.init_ordering_mode == 'random_onehot':
            while True:
                initialized_modules = set()
                for t in range(self.num_init_tasks):
                    modules = np.random.randint(self.depth, size=self.depth)
                    self.structure[t].data = -np.inf * \
                        torch.ones(self.depth, self.depth)
                    self.structure[t].data[modules, np.arange(self.depth)] = 1
                    for m in modules:
                        initialized_modules.add(m)
                if len(initialized_modules) == self.depth:
                    break
        elif self.init_ordering_mode == 'random':
            raise NotImplementedError
        elif self.init_ordering_mode == 'uniform':
            pass
        else:
            raise ValueError('{} is not a valid ordering initialization mode'.format(
                self.init_ordering_mode))

    def freeze_structure(self):
        self.freeze_linear_weights()
        self.freeze_decoder()

    def unfreeze_structure(self, task_id):
        self.unfreeze_linear_weights(task_id)
        self.unfreeze_decoder(task_id)

    def freeze_linear_weights(self):
        for param in self.structure:
            param.requires_grad = False
            param.grad = None

    def unfreeze_linear_weights(self, task_id):
        self.structure[task_id].requires_grad = True

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
            param.grad = None

    def unfreeze_decoder(self, task_id):
        for param in self.decoder[task_id].parameters():
            param.requires_grad = True
