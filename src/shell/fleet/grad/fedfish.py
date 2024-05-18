from shell.fleet.grad.monograd import *
from shell.fleet.grad.fedcurv_utils import EWC
import copy
import torch
import ray
from collections import defaultdict

class FedFishAgent(ModelSyncAgent):
    def __init__(self, node_id: int, seed: int, dataset, NetCls, AgentCls, net_kwargs, agent_kwargs, train_kwargs, sharing_strategy):
        if "fl_strategy" not in agent_kwargs:
            agent_kwargs["fl_strategy"] = "fedfish"
        super().__init__(node_id, seed, dataset, NetCls, AgentCls,
                            net_kwargs, agent_kwargs, train_kwargs, sharing_strategy)

    def train(self, task_id, start_epoch=0, communication_frequency=None, final=True):
        self.agent.mu = self.sharing_strategy.mu
        return super().train(task_id, start_epoch, communication_frequency, final)

    def prepare_communicate(self, task_id, end_epoch, comm_freq, num_epochs, communication_round, final=False):
        super().prepare_communicate(task_id, end_epoch, comm_freq, num_epochs, communication_round, final)

        tmp_dataset = copy.deepcopy(self.dataset.trainset[task_id])
        tmp_dataset.tensors = tmp_dataset.tensors + (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        mega_dataset = ConcatDataset(
            [get_custom_tensordataset(replay.get_tensors(), name=self.dataset.name, use_contrastive=self.agent.use_contrastive) for replay in self.agent.replay_buffers.values()] +
            [tmp_dataset] +
            [get_custom_tensordataset(replay.get_tensors(), name=self.dataset.name, use_contrastive=self.agent.use_contrastive) for replay in self.agent.shared_replay_buffers.values() if len(replay) > 0]
        )
        mega_loader = torch.utils.data.DataLoader(mega_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        temperature = getattr(self.sharing_strategy, "temperature", 1.0)
        self.fisher = EWC(self.agent, mega_loader, normalize=True, temperature=temperature).fisher

    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            neighbor.receive(self.node_id, deepcopy(self.model), "model")

    def receive(self, node_id, model, msg_type):
        if msg_type == "model":
            self.incoming_models[node_id] = model
        else:
            raise ValueError(f"Invalid message type: {msg_type}")

    def train(self, task_id, start_epoch=0, communication_frequency=None, final=True):
        self.agent.incoming_models = self.incoming_models
        self.agent.mu = self.sharing_strategy.mu
        return super().train(task_id, start_epoch, communication_frequency, final)

    def aggregate_models(self):
        if len(self.incoming_models.values()) == 0:
            return
        logging.info("Fedfish AGGREGATING MODELS...no_components %s", len(self.net.components))
        # Compute the average of the incoming models including the current node's model
        avg_model = {name: torch.zeros_like(param) for name, param in self.model.items()}
        num_models = len(self.incoming_models) + 1  # +1 for the current node's model

        for model in self.incoming_models.values():
            for name, param in model.items():
                avg_model[name] += param.data

        # Include the current node's model
        for name, param in self.model.items():
            avg_model[name] += param.data
        for name, param in avg_model.items():
            avg_model[name] /= num_models
            # logging.info("avg_model %s %s", name, param)

        # Apply the FedFish aggregation
        for name, param in self.model.items():
            diag_fisher = self.fisher[name].clamp(0, 1)  # Ensure Fisher diagonal is between 0 and 1
            self.net.state_dict()[name].data.copy_(diag_fisher * param.data + (1 - diag_fisher) * avg_model[name])
            # self.net.state_dict()[name].data.copy_(avg_model[name])

@ray.remote
class ParallelFedFishAgent(FedFishAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(self.node_id, deepcopy(self.model), "model"))


class FedFishModAgent(FedFishAgent):
    def prepare_model(self):
        num_init_components = self.net.depth
        num_components = len(self.net.components)
        for i in range(num_init_components, num_components):
            self.excluded_params.add("components.{}".format(i))
        return super().prepare_model()


@ray.remote
class ParallelFedFishModAgent(FedFishModAgent):
    def communicate(self, task_id, communication_round, final=False):
        for neighbor in self.neighbors.values():
            ray.get(neighbor.receive.remote(
                self.node_id, deepcopy(self.model), "model"))
