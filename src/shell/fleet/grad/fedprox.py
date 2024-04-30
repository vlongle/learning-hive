from shell.fleet.grad.monograd import *
import copy


class FedProxAgent(ModelSyncAgent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs,
                 sharing_strategy, agent=None):
        if "fl_strategy" not in agent_kwargs:
            agent_kwargs["fl_strategy"] = "fedprox"
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy, agent=agent)

    def process_communicate(self, task_id, communication_round, final=False):
        self.agent.global_model = copy.deepcopy(self.agent.net)
        self.agent.mu = self.sharing_strategy.mu
        self.agent.excluded_params = self.excluded_params
        return super().process_communicate(task_id, communication_round, final)


@ray.remote
class ParallelFedProxAgent(FedProxAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))


class FedProxModAgent(FedProxAgent):
    def prepare_model(self):
        num_init_components = self.net.depth
        num_components = len(self.net.components)
        for i in range(num_init_components, num_components):
            self.excluded_params.add("components.{}".format(i))
        return super().prepare_model()


@ray.remote
class ParallelFedProxModAgent(FedProxModAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
