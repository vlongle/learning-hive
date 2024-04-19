from shell.fleet.grad.monograd import *
import copy


class FedProxAgent(ModelSyncAgent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        if "fl_strategy" not in agent_kwargs:
            agent_kwargs["fl_strategy"] = "fedprox"
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True):
        self.agent.global_model = copy.deepcopy(self.agent.net)
        self.agent.mu = self.sharing_strategy.mu
        return super().train(task_id, start_epoch, communication_frequency, final)


@ray.remote
class ParallelFedProxAgent(FedProxAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
