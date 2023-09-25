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
        pass


    def prepare_communicate(self, task_id, communication_round):
        self.outgoing_modules = {}
        for neighbor in self.neighbors:
            self.outgoing_modules[neighbor.node_id] = self.select_module(neighbor.node_id)

    def receive(self, sender_id, data, msg_type):
        self.incoming_modules = {}
        self.incoming_modules[sender_id] = data


    def communicate(self, task_id, communication_round):
        for neighbor in self.neighbors:
            neighbor.receive(self.node_id, self.outgoing_modules[neighbor.node_id], "module")

    def process_communicate(self, task_id, communication_round):
        pass