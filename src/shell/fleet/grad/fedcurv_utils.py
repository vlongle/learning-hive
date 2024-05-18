import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from copy import deepcopy
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from torch.nn import Linear, Module


class EWC(object):
    def __init__(self, model: nn.Module, dataloader, normalize=False, temperature=1.0):

        self.model = model
        self.dataloader = dataloader
        self._means = {}
        self.normalize_fisher = normalize
        self.temperature = temperature
        self.fisher = self._diag_fisher()

    def _diag_fisher(self):
        self.model.net.eval()
        importances = {}

        # Convert generator to a dict and then deepcopy
        params = {name: param for name,
                  param in self.model.net.named_parameters()}
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
                l_t = self.model.compute_loss(Xt,
                                              Yt, task_id_tmp,
                                              mode='ce',
                                              global_step=None,
                                              use_aux=False,
                                              )
                l += l_t

            if l is None or not l.requires_grad:
                continue  # for modular, in the rare case that this batch
                # doesn't contain the current task. Skip it.
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


        if self.normalize_fisher:
            importances = self._normalize_importances(importances)

        self.model.net.train()

        return importances


    def _normalize_importances(self, importances):
        # Flatten all importances into a single tensor
        all_importances = torch.cat([imp.view(-1) for imp in importances.values()])
        # print('>>> MIN:', torch.min(all_importances), 'MAX:', torch.max(all_importances), 'MEAN:', torch.mean(all_importances), 'std:', torch.std(all_importances))

        # Normalize the importances before applying softmax
        mean_importances = torch.mean(all_importances)
        std_importances = torch.std(all_importances)
        normalized_importances = (all_importances - mean_importances) / std_importances

        # Apply softmax to the normalized tensor
        softmax_importances = torch.softmax(normalized_importances / self.temperature, dim=0)

        # print('>>> AFTERWARDS  MIN:', torch.min(softmax_importances), 'MAX:', torch.max(softmax_importances), 'MEAN:', torch.mean(softmax_importances), 'std:', torch.std(softmax_importances))
        # Assign the normalized values back to their original shape
        start = 0
        for k, imp in importances.items():
            num_elements = imp.numel()
            normalized_imp = softmax_importances[start:start + num_elements].view(imp.shape)
            importances[k] = normalized_imp
            start += num_elements
        
        # Final check to ensure all importances are between 0 and 1 after assignment
        for k, imp in importances.items():
            assert torch.all(imp >= 0) and torch.all(imp <= 1), f"Normalized importances for {k} are not in the range [0, 1]"
            # if torch.any(imp > 0) and torch.unique(imp).numel() > 1:
            #     print("k", k, "imp", imp)
            #     exit(0)
        
        return importances



