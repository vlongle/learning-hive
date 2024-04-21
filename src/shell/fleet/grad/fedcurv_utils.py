import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from copy import deepcopy
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from torch.nn import Linear, Module




class EWC(object):
    def __init__(self, model: nn.Module, dataloader):

        self.model = model
        self.dataloader = dataloader
        self._means = {}
        self.fisher = self._diag_fisher()


    def _diag_fisher(self):
        self.model.net.eval()
        importances = {}


        # Convert generator to a dict and then deepcopy
        params = {name: param for name, param in self.model.net.named_parameters()}
        copied_params = deepcopy(params)

        for n, p in copied_params.items():
            p.data.zero_()
            importances[n] = p.data.clone()

        for X, Y, t in self.dataloader:
            # get only input, target and task_id from the batch
            X = X.to(self.model.net.device, non_blocking=True)
            Y = Y.to(self.model.net.device, non_blocking=True)

            self.model.optimizer.zero_grad()

            l = 0.
            for task_id_tmp in torch.unique(t):
                # for task_id_tmp in sorted(all_t.tolist(), reverse=True):
                Yt = Y[t == task_id_tmp]
                Xt = X[t == task_id_tmp]
                l += self.model.compute_loss(Xt,
                                             Yt, task_id_tmp,
                                             mode=None,
                                             global_step=None,
                                             use_aux=False,
                                             )

            l.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.net.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(self.dataloader))

        self.model.net.train()

        return importances

    # def penalty(self, model: nn.Module):
    #     loss = 0
    #     for n, p in model.named_parameters():
    #         _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
    #         loss += _loss.sum()
    #     return loss
