'''
File: /fisher_utils.py
Project: grad
Created Date: Thursday September 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
'''
File: /fisher.py
Project: experiments
Created Date: Wednesday September 6th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
class EWC(object):
    def __init__(self, model: nn.Module, dataloaders, device="cuda"):
        """
        dataloaders: dict of various tasks
        """

        self.model = model
        self.dataloaders = dataloaders
        self.tot_n_pts = sum([len(dataloader.dataset)
                              for dataloader in self.dataloaders.values()])
        self.device = device

        self.params = {n: p for n,
                       p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.to(self.device)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(self.device)

        self.model.eval()
        for task_id, dataloader in self.dataloaders.items():
            for input, target, _ in dataloader:
                input, target = input.to(self.device), target.to(self.device)

                self.model.zero_grad()
                output = self.model(input, task_id)
                # TODO: this loss might be changed in the case of contrastive
                loss = F.nll_loss(F.log_softmax(output, dim=1), target)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if n in precision_matrices and p.grad is not None:
                        precision_matrices[n] += p.grad.data ** 2 / \
                            self.tot_n_pts

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim.Optimizer, dataloader, device):
    model.train()
    epoch_loss = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def ewc_train(model: nn.Module, optimizer: torch.optim.Optimizer, dataloader, ewc: EWC, importance: float, device):
    model.train()
    epoch_loss = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + \
            importance * ewc.penalty(model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def test(model: nn.Module, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            correct += (F.softmax(output, dim=1).max(dim=1)
                        [1] == target).sum().item()
    return correct / len(dataloader.dataset)
