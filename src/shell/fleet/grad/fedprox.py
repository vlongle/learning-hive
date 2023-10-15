from shell.fleet.grad.monograd import *
import copy

class FedProxAgent(ModelSyncAgent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        agent_kwargs["fl_strategy"] = "fedprox"
        super().__init__(node_id, seed, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
    def train(self, task_id, start_epoch=0, communication_frequency=None,
              final=True):
        self.agent.global_model = copy.deepcopy(self.agent.net)
        self.agent.mu = self.sharing_strategy.mu
        return super().train(task_id, start_epoch, communication_frequency, final)
@ray.remote
class ParallelFedProxAgent(FedProxAgent):
    def communicate(self, task_id, communication_round):
        # logging.info(
        #     f"node {self.node_id} is communicating at round {communication_round} for task {task_id}")
        # TODO: Should we do deepcopy???
        # put model on object store
        # state_dict = deepcopy(self.net.state_dict())
        # model = state_dict
        # model = ray.put(state_dict)
        # send model to neighbors
        # logging.info(f"My neighbors are: {self.neighbors}")
        for neighbor in self.neighbors:
            # neighbor_id = ray.get(neighbor.get_node_id.remote())
            # NOTE: neighbor_id for some reason is NOT responding...
            # logging.info(f"SENDING MODEL: {self.node_id} -> {neighbor_id}")
            ray.get(neighbor.receive.remote(self.node_id, self.model, "model"))
            self.bytes_sent[(task_id, communication_round)
                            ] = self.compute_model_size(self.model)
