'''
File: /fisher_modgrad.py
Project: grad
Created Date: Thursday September 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


from shell.fleet.grad.modgrad import ModGrad


class FisherModGrad(ModGrad):
    def prepare_communicate(self, task_id, communication_round):
        """
        TODO: precompute the fisher diagonal weighting
        """
        pass

    def aggregate_models(self):
        """
        TODO: change the aggregation scheme to make sure of the fisher information
        """
        pass
