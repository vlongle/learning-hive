'''
File: /test.py
Project: learning-hive
Created Date: Thursday March 23rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import hydra
from omegaconf import DictConfig

from shell.fleet.utils.fleet_utils import get_fleet, get_agent_cls
import torch
import time
import datetime
import logging
from shell.utils.experiment_utils import setup_experiment, process_dataset_cfg
from shell.datasets.datasets import get_dataset
from pprint import pprint
import ray
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="conf", config_name="grad", version_base=None)
def main(cfg: DictConfig) -> None:
    pprint(cfg)
    # cfg.parallel = False
    cfg.parallel = True
    cfg.train.num_epochs = 10
    cfg.train.component_update_freq = 10
    cfg.num_agents = 4
    AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.parallel)

    graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg = setup_experiment(
        cfg)

    FleetCls = get_fleet(cfg.sharing_strategy, cfg.parallel)

    # create one extra fake dataset for testing
    fake_dataset = get_dataset(**process_dataset_cfg(cfg))
    fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                     LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                     train_kwargs=train_cfg, fake_dataset=fake_dataset)

    # check that all the agents now have the same model!
    # and the structures of all agents = 1 since we already do the pretraining...

    for agent in fleet.agents:
        # check that the random_linear_projection is the same. use torch.allclose
        # for weight
        if cfg.parallel:
            net1 = ray.get(agent.get_net.remote())
            net2 = ray.get(fleet.agents[0].get_net.remote())
        else:
            net1 = agent.get_net()
            net2 = fleet.agents[0].get_net()
        assert torch.allclose(net1.random_linear_projection.weight,
                              net2.random_linear_projection.weight)

    # components are the same
    for agent in fleet.agents:
        if cfg.parallel:
            num_components = ray.get(agent.get_num_components.remote())
        else:
            num_components = agent.get_num_components()
        for component_id in range(num_components):
            if cfg.parallel:
                net1 = ray.get(agent.get_net.remote())
                net2 = ray.get(fleet.agents[0].get_net.remote())
            else:
                net1 = agent.get_net()
                net2 = fleet.agents[0].get_net()
            assert torch.allclose(net1.components[component_id].weight,
                                  net2.components[component_id].weight)

    # check that the decoder are NOT the same
    for i in range(1, len(fleet.agents)):
        for task in range(cfg.dataset.num_tasks):
            if cfg.parallel:
                net1 = ray.get(fleet.agents[i].get_net.remote())
                net2 = ray.get(fleet.agents[i-1].get_net.remote())
            else:
                net1 = fleet.agents[i].get_net()
                net2 = fleet.agents[i-1].get_net()
            assert not torch.allclose(net1.decoder[task].weight,
                                      net2.decoder[task].weight)
    # check that structures for future tasks are torch.ones
    # if net has structure
    if cfg.parallel:
        net1 = ray.get(fleet.agents[0].get_net.remote())
    else:
        net1 = fleet.agents[0].get_net()
    if hasattr(net1, "structure"):
        for agent in fleet.agents:
            for task in range(cfg.dataset.num_tasks):
                if cfg.parallel:
                    net1 = ray.get(agent.get_net.remote())
                else:
                    net1 = agent.get_net()
                assert torch.allclose(net1.structure[task],
                                      torch.ones_like(net1.structure[task]))
    # TODO: check that the performance on fake_data is good!

    print("All tests passed!")


if __name__ == "__main__":
    main()
