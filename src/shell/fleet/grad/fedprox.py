
import torch
from shell.fleet.grad.monograd import ModelSyncAgent


class ModelProxAgent(ModelSyncAgent):
    
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                         net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)
        self.proximal_term = None
        
    def aggregate_models(self):
        super().aggregate_models()
        
        # take l2 norm of (this_model_weights - other_model_weights_k)
        
        for model in self.incoming_models.values():
            for name, param in model.items():
                diff = param.data - self.model[name]
                l2_norm = torch.norm(diff, 2)
                self.proximal_term += l2_norm ** 2
