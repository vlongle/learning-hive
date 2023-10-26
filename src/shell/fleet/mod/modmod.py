'''
File: /modmod.py
Project: fleet
Created Date: Tuesday March 28th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


from shell.fleet.fleet import Agent


class ModModAgent(Agent):
    def select_module(self, neighbor_id):
        outgoing_modules = []
        if self.sharing_strategy.module_selection == "naive":
            # send all the new modules
            num_newly_added_modules = len(self.net.components) - self.net.depth - len(self.net.candidate_indices)
            outgoing_modules = []
            for i in range(num_newly_added_modules):
                assert i not in self.net.candidate_indices
                outgoing_modules.append(self.net.components[i])
        else:
            raise NotImplementedError(f"Module selection {self.sharing_strategy.module_selection} not implemented.")
        
        return outgoing_modules


    def prepare_communicate(self, task_id, communication_round):
        self.outgoing_modules = {}
        self.incoming_modules = {}
        for neighbor in self.neighbors:
            self.outgoing_modules[neighbor.node_id] = self.select_module(neighbor.node_id)

    def receive(self, sender_id, data, msg_type):
        self.incoming_modules[sender_id] = data


    def communicate(self, task_id, communication_round):
        for neighbor in self.neighbors:
            neighbor.receive(self.node_id, self.outgoing_modules[neighbor.node_id], "module")

    def process_communicate(self, task_id, communication_round):
        module_list = []
        for neighbor in self.neighbors:
            module_list+= self.incoming_modules[neighbor.node_id]
        self.train_kwargs["module_list"] = module_list 