'''
File: /modmod.py
Project: fleet
Created Date: Tuesday March 28th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


from shell.fleet.fleet import Agent
from shell.fleet.data.data_utilize import compute_tasks_sim
import ray

# NOTE: HACK: we ignore sharing in the first num_init_tasks + 1 tasks
# because there's no dynamically added module yet in these. Although
# for sync=False, each agent trained their base modules separately. In
# princeiple, we could have pick the module that is most activated
# for a given task to send (e.g., by looking at the softmax structure)


class ModModAgent(Agent):
    def compute_task_similarity(self, neighbor_task, task_id):
        candidate_tasks = [self.dataset.class_sequence[t *
                                                       self.dataset.num_classes_per_task: (t+1) * self.dataset.num_classes_per_task] for t in range(task_id + 1)]
        return [compute_tasks_sim(task, neighbor_task) for task in candidate_tasks]

    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True, **kwargs):

        module_list = self.train_kwargs.get("module_list", [])
        if self.sharing_strategy.opt_with_random:
            # optimize the received module along with a random module
            # as well
            num_candidate_modules = len(module_list) + 1
        else:
            num_candidate_modules = len(module_list)

        if len(module_list) == 0:
            num_candidate_modules = 1  # at the very least, we need to consider a random module

        if "num_candidate_modules" not in kwargs:  # HACK: to be compatible with the exploratory ipynb
            kwargs["num_candidate_modules"] = num_candidate_modules

        train_candidate_module = not self.sharing_strategy.freeze_candidate_module

        return super().train(task_id, start_epoch, communication_frequency, final, train_candidate_module=train_candidate_module,
                             **kwargs)

    def select_module(self, neighbor_id, task_id):
        outgoing_modules = {}
        if self.sharing_strategy.module_selection == "naive":
            # send all the new modules
            num_newly_added_modules = len(
                self.net.components) - self.net.depth - len(self.net.candidate_indices)
            outgoing_modules = []
            for i in range(num_newly_added_modules):
                assert i not in self.net.candidate_indices
                outgoing_modules.append(self.net.components[i])
        elif self.sharing_strategy.module_selection == "gt_most_similar":
            outgoing_modules = self.send_most_similar_module(
                neighbor_id, task_id)
        else:
            raise NotImplementedError(
                f"Module selection {self.sharing_strategy.module_selection} not implemented.")

        return outgoing_modules

    def send_most_similar_module(self, neighbor_id, task_id):
        task_sims = self.task_sims[neighbor_id]
        module_record = self.agent.dynamic_record.df

        for t in range(len(task_sims)):
            if t not in set(module_record['task_id']):
                task_sims[t] = 0
            if t in set(module_record['task_id']) and not module_record[module_record['task_id'] == t]['add_new_module'].item():
                task_sims[t] = 0

        # get the most similar task with the highest similarity. Break ties by the task id
        # (highest wins)
        # most_similar_task = max(
        #     range(len(task_sims)), key=lambda x: (task_sims[x], x))
        # lowest task_id wins
        most_similar_task = max(
            range(len(task_sims)), key=lambda x: (task_sims[x], -x))
        if task_sims[most_similar_task] == 0:
            return []

        task_module = module_record[module_record['task_id']
                                    == most_similar_task]['num_components'].item() - 1
        # print('node', self.node_id, 'for neighbor', neighbor_id, '@ task', task_id, 'sending module', task_module, 'from task', most_similar_task,
        #       'with similarity', task_sims[most_similar_task], 'current no. of modules', len(self.net.components))
        # pathological for replaying ipynb
        if task_module >= len(self.net.components):
            return []
        return [{'task_id': most_similar_task, 'task_sim': task_sims[most_similar_task],
                 'module_id': task_module, 'module': self.net.components[task_module]}]

    def prepare_communicate(self,  task_id, end_epoch, comm_freq, num_epochs, communication_round,
                            final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 1:
            self.outgoing_modules = {}
            self.incoming_modules = {}
            for neighbor_id, neighbor in self.neighbors.items():
                self.outgoing_modules[neighbor_id] = self.select_module(
                    neighbor_id, task_id)
        else:
            self.query_tasks = {}

    def receive(self, sender_id, data, msg_type):
        if msg_type == "query_task":
            self.query_tasks[sender_id] = data
        elif msg_type == "module":
            self.incoming_modules[sender_id] = data

    def communicate(self, task_id, communication_round, final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 0:
            self.send_query_task(task_id)
        else:
            for neighbor_id, neighbor in self.neighbors.items():
                neighbor.receive(
                    self.node_id, self.outgoing_modules[neighbor_id], "module")

    def choose_best_module_from_neighbors(self, module_list):
        # module_list is a list of (t, sim, module)
        # find the most similar task based on the similarity score. Break ties by the task id
        # (highest task id wins)
        # best_match = max(module_list, key=lambda x: (
        #     x[1], x[0]))
        # lowest task_id wins
        best_match_index = max(enumerate(module_list),
                               key=lambda x: (x[1]['task_sim'], -x[1]['task_id']))[0]
        return best_match_index

    def get_module_list(self):
        module_list = []
        for neighbor_id in self.neighbors:
            extra_info_ls = [e | {'neighbor_id': neighbor_id}
                             for e in self.incoming_modules[neighbor_id]]
            module_list += extra_info_ls
        return module_list

    def process_communicate(self, task_id, communication_round, final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 1:
            module_list = self.get_module_list()
            if len(module_list) == 0:
                self.train_kwargs["module_list"] = []
                return

            best_match = module_list[self.choose_best_module_from_neighbors(
                module_list)]
            self.train_kwargs["module_list"] = [best_match['module']]
        else:
            self.task_sims = {}
            for neighbor_id in self.neighbors:
                neighbor_task = self.query_tasks[neighbor_id]
                self.task_sims[neighbor_id] = self.compute_task_similarity(
                    neighbor_task, task_id)

    def send_query_task(self, task_id):
        task = self.dataset.class_sequence[task_id *
                                           self.dataset.num_classes_per_task: (task_id + 1) * self.dataset.num_classes_per_task]
        for neighbor in self.neighbors.values():
            neighbor.receive(self.node_id, task, "query_task")


@ray.remote
class ParallelModModAgent(ModModAgent):
    def send_query_task(self, task_id):
        task = self.dataset.class_sequence[task_id *
                                           self.dataset.num_classes_per_task: (task_id + 1) * self.dataset.num_classes_per_task]
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(self.node_id, task, "query_task"))

    def communicate(self, task_id, communication_round, final=None):
        if task_id < self.net.num_init_tasks + 1:
            return
        if communication_round % 2 == 0:
            self.send_query_task(task_id)
        else:
            for neighbor_id, neighbor in self.neighbors.items():
                ray.get(neighbor.receive.remote(
                    self.node_id, self.outgoing_modules[neighbor_id], "module"))
