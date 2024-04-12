from shell.fleet.fleet import Agent
import torch
from shell.learners.base_learning_classes import CompositionalDynamicLearner
from shell.utils.replay_buffers import ReplayBufferReservoir
import numpy as np
import ray
from shell.fleet.data.data_utilize import get_local_label_for_task


def adjust_allocation(query, budget):
    """
    Adjusts the allocation of instances to integer values while ensuring the sum equals the budget.

    Parameters:
    - query: A dictionary with class as key and the allocated float number as value.
    - budget: Total number of instances to distribute.

    Returns:
    - A dictionary with class as key and integer number of instances to send as value.
    """
    # Sort classes by the fractional part of the allocation in descending order
    # This helps in deciding which classes to round up.
    sorted_classes = sorted(
        query, key=lambda x: query[x] - np.floor(query[x]), reverse=True)

    # Initialize the adjusted query with floor values of allocations
    adjusted_query = {cls: int(np.floor(allocation))
                      for cls, allocation in query.items()}

    # Distribute remaining budget (due to flooring) by rounding up allocations in sorted order
    remaining_budget = budget - sum(adjusted_query.values())
    for cls in sorted_classes[:remaining_budget]:
        adjusted_query[cls] += 1

    return adjusted_query


class HeuristicDataAgent(Agent):

    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs,
                 sharing_strategy):
        # self.use_ood_separation_loss = sharing_strategy.use_ood_separation_loss
        # agent_kwargs['use_ood_separation_loss'] = self.use_ood_separation_loss
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

        self.is_modular = isinstance(self.agent, CompositionalDynamicLearner)
        self.min_task = getattr(self.sharing_strategy, 'min_task', 0)

    @torch.inference_mode()
    def compute_query(self, task_id, mode="all"):
        """
        Compute query using a validation

        If mode="all", get the query for all tasks up to `task_id`,
        If mode="current", get the query for the current task only.

        Rank all instances across all tasks together based on their scores,
        and then select the top `num_queries` instances in a vectorized manner.
        """

        was_training = self.net.training
        self.net.eval()
        if mode == "all":
            tasks = range(task_id + 1)
        elif mode == "current":
            tasks = [task_id]
        else:
            raise ValueError(f"Invalid mode {mode}")

        test_perf = self.eval_val(tasks)

        test_perf = {
            tuple(self.get_task_class(t)): v for t, v in test_perf.items()
        }

        # self.get_task_class(t) is a tuple of class labels for task t
        # we now need to flatten test_perf such that we have test_perf[class] = score
        test_perf = {
            c: v for t, v in test_perf.items() for c in t
        }

        data_worth = {
            c: 1.0 - v for c, v in test_perf.items()
        }

        if was_training:
            # print('COMPUTE QUERY SETTING TRAINING AGAIN')
            self.net.train()

        return data_worth

    def prepare_communicate(self, task_id, end_epoch, comm_freq, num_epochs, communication_round, final=False,):
        if task_id < self.agent.net.num_init_tasks:
            return

        if communication_round % 2 == 0:
            self.incoming_query, self.incoming_data = {}, {}
            if 'query_task_mode' not in self.sharing_strategy:
                mode = "all"
                if self.is_modular:
                    component_update_freq = self.train_kwargs['component_update_freq']
                    next_end_epoch = min(end_epoch + comm_freq, num_epochs)
                    has_comp_update = component_update_freq is not None and next_end_epoch % component_update_freq == 0
                    if not has_comp_update:
                        mode = "current"
            else:
                mode = self.sharing_strategy['query_task_mode']
            self.query = self.compute_query(task_id, mode=mode)
        else:
            self.compute_data(task_id)

    def compute_data(self, task_id):
        available_data = self.get_data_pool(task_id)
        self.data = {}
        for requester, query in self.incoming_query.items():
            self.data[requester] = self.get_data(
                available_data, query, task_id)

    def get_candidate_data(self, task, map_to_global=True):
        if task == self.agent.T - 1:
            Xt, yt = self.dataset.trainset[task].tensors
        else:
            Xt, yt, _ = self.agent.replay_buffers[task].get_tensors()
        if map_to_global:
            # map yt back to the global class labels
            task_cls = self.get_task_class(task)
            task_cls_tensor = torch.tensor(task_cls, dtype=torch.long)
            yt = task_cls_tensor[yt]
        return Xt, yt

    def get_data_pool(self, task_id):
        available_data = {}
        for t in range(self.min_task, task_id + 1):
            Xt, yt = self.get_candidate_data(t)
            for c in yt.unique():
                if c.item() not in available_data:
                    available_data[c.item()] = []
                available_data[c.item()].append(Xt[yt == c])
        return available_data

    def get_data(self, available_data, query, task_id):
        available_cls = set(self.dataset.class_sequence[:(
            task_id + 1) * self.dataset.num_classes_per_task])
        query = {k: v for k, v in query.items() if k in available_cls}

        total_weight = sum(query.values())

        probabilities = {cls: weight /
                         total_weight for cls, weight in query.items()}

        # Allocate budget based on probabilities
        budget_allocation = {cls: self.sharing_strategy.budget *
                             prob for cls, prob in probabilities.items()}

        # Adjust allocation to ensure integer counts and sum equals budget
        budget_allocation = adjust_allocation(
            budget_allocation, self.sharing_strategy.budget)

        # Initialize a dictionary to hold the data to send
        data_to_send = {}

        # For each class in the budget allocation, select instances to send
        for cls, num_instances in budget_allocation.items():
            if cls in available_data:
                # Get all available instances for the class
                instances = torch.cat(available_data[cls], dim=0)
                # If there are more available instances than the allocated number, select randomly
                if len(instances) > num_instances:
                    selected_instances = instances[np.random.choice(
                        len(instances), size=num_instances, replace=False)]
                else:
                    # If allocation matches available instances, send all
                    selected_instances = instances
                # Add selected instances to the data to send
                data_to_send[cls] = selected_instances

        return data_to_send

    def get_consolidated_data(self):
        consolidated_data = {}
        # Iterate over incoming data from all neighbors
        for neighbor_id, neighbor_data in self.incoming_data.items():
            # Iterate over each class in the neighbor's data
            for cls, instances in neighbor_data.items():
                # If the class is not already in the consolidated data, add it
                if cls not in consolidated_data:
                    consolidated_data[cls] = []
                # Add the neighbor's instances for the class to the consolidated data
                # This example simply appends instances; consider checking for and removing duplicates if needed
                consolidated_data[cls].append(instances)

        consolidated_data = {cls: torch.cat(
            instances, dim=0) for cls, instances in consolidated_data.items()}
        return consolidated_data

    def add_incoming_data(self, task_id):
        consolidated_data = self.get_consolidated_data()

        for t in range(task_id+1):
            task_cls = self.get_task_class(t)
            data = {c: consolidated_data[c]
                    for c in task_cls if c in consolidated_data}
            data.update({c: torch.empty(0) for c in task_cls if c not in data})
            if self.sharing_strategy.enforce_balance:
                min_size = min(data[c].size(0) for c in data) if data else 0
                # Truncate each class's data to the minimum size
                data = {c: instances[:min_size]
                        for c, instances in data.items()}

            if t not in self.agent.shared_replay_buffers:
                self.agent.shared_replay_buffers[t] = ReplayBufferReservoir(
                    self.sharing_strategy.shared_memory_size, task_id, self.sharing_strategy.hash_data)

            for c, X in data.items():
                local_c = get_local_label_for_task(c.item(), t, self.dataset.class_sequence,
                                                   self.dataset.num_classes_per_task)
                Y = torch.tensor([local_c] * X.size(0))
                if len(Y) == 0:
                    continue
                self.agent.shared_replay_buffers[t].push(X, Y)

    def communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks:
            # NOTE: don't communicate for the first few tasks to
            # allow agents some initital training to find their weakness
            return
        if communication_round % 2 == 0:
            # send query to neighbors
            for neighbor in self.neighbors.values():
                neighbor.receive(self.node_id, self.query, "query")
        else:
            # send data to the requester
            # for requester in self.incoming_query:
            #     self.neighbors[requester].receive(
            #         self.node_id, self.data[requester], "data")
            for neighbor_id, neighbor in self.neighbors.items():
                neighbor.receive(
                    self.node_id, self.data[neighbor_id], "data")

    def receive(self, sender_id, data, data_type):
        if data_type == "query":
            # add query to buffer
            self.incoming_query[sender_id] = data
        elif data_type == "data":
            # add data to buffer
            self.incoming_data[sender_id] = data
        else:
            raise ValueError("Invalid data type")

    def process_communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks:
            return
        if communication_round % 2 == 0:
            pass
        else:
            self.add_incoming_data(task_id)


@ray.remote
class ParallelHeuristicDataAgent(HeuristicDataAgent):
    def communicate(self, task_id, communication_round, final=False):
        if task_id < self.agent.net.num_init_tasks:
            # NOTE: don't communicate for the first few tasks to
            # allow agents some initital training to find their weakness
            return
        if communication_round % 2 == 0:
            # send query to neighbors
            for neighbor in self.neighbors.values():
                ray.get(neighbor.receive.remote(
                    self.node_id, self.query, "query"))
        else:
            # send data to the requester
            # for requester in self.incoming_query:
            #     self.neighbors[requester].receive(
            #         self.node_id, self.data[requester], "data")
            for neighbor_id, neighbor in self.neighbors.items():
                ray.get(neighbor.receive.remote(
                    self.node_id, self.data[neighbor_id], "data"))
