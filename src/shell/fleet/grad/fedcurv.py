import torch
from shell.fleet.grad.monograd import ModelSyncAgent
from shell.fleet.grad.fisher_utils import EWC   


class ModelProxAgent(ModelSyncAgent):
    
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
        
    def prepare_fisher_diag(self):
        ewc = EWC(self.model, self.agent.memory_loaders)
        return ewc._precision_matrices
        
    def prepare_communicate(self, task_id, communication_round):
        self.model
        self._fisher_diag = self.prepare_fisher_diag()  # normalize to 0 and 1