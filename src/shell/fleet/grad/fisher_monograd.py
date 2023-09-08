'''
File: /fishergrad.py
Project: grad
Created Date: Thursday September 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

from shell.fleet.grad.monograd import ModelSyncAgent
from collections import defaultdict
from shell.fleet.grad.fisher_utils import EWC


class ModelFisherSyncAgent(ModelSyncAgent):
    def prepare_fisher_diag(self):
        # compute the fisher on the replayed data
        ewc = EWC(self.model, self.agent.memory_loaders)
        return ewc._precision_matrices

    def prepare_communicate(self, task_id, communication_round):
        """
        TODO: precompute the fisher diagonal weighting
        """
        self.model = self.prepare_model()
        self._fisher_diag = self.prepare_fisher_diag()  # normalize to 0 and 1

    def aggregate_models(self):
        """
        TODO: change the aggregation scheme to make sure of the fisher information
        """
        # get model from neighbors
        # average all the models together!
        # stuff_added = defaultdict(int)
        for name, param in self.net.state_dict().items():
            self.net.state_dict()[name].data *= self._fisher_diag[name]

        for model in self.incoming_models.values():
            for name, param in model.items():
                # print("Adding name:", name)
                self.net.state_dict()[name].data += (param.data) * \
                    (1-self._fisher_diag[name])
                # stuff_added[name] += 1

        for name, param in self.net.state_dict().itenms():
            # self.net.state_dict()[name].data /= len(self.incoming_models) + 1
            self.net.state_dict()[name].data /= len(self.incoming_models)

        # normalize
        # for name, param in self.net.state_dict().items():
            # +1 because it includes the current model
            # NOTE: not sure if this is correct actually
            # param.data /= stuff_added[name] + 1
