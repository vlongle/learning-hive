'''
File: /test_fisher_grad.py
Project: experiments
Created Date: Thursday September 7th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import hydra
from omegaconf import DictConfig

from shell.fleet.utils.fleet_utils import get_fleet, get_agent_cls
from shell.fleet.grad.fisher_monograd import ModelFisherSyncAgent

import time
import datetime
import logging
from shell.utils.experiment_utils import setup_experiment
logging.basicConfig(level=logging.INFO)


num_init_tasks = 4
cfg = {
    "algo": "monolithic",
    "seed": 0,
    "parallel": True,
    "num_agents": 2,
    "dataset": "mnist",
    "dataset.num_trains_per_class": 64,
    "dataset.num_vals_per_class": 50,
    "dataset.remap_labels": True,
    "dataset.with_replacement": True,
    "dataset.num_tasks": 10-num_init_tasks,  # NOTE: we already jointly
    # train using a fake agent.
    "net": "mlp",
    "net.depth": 4,
    "num_init_tasks": num_init_tasks,
    "net.dropout": 0.0,
    "train.num_epochs": 100,
    "train.component_update_freq": 100,
    "train.init_num_epochs": 100,
    "train.init_component_update_freq": 100,
    "train.save_freq": 1,
    "agent.use_contrastive": True,
    "agent.memory_size": 32,
    # "dataset": ["mnist", "kmnist", "fashionmnist"],
    "dataset": "mnist",
    # "root_save_dir": "grad_new_unfreeze_all_decoders_retrain_results",
    "root_save_dir": "test_fisher_monograd_results",
    # ================================================
    # GRAD SHARING SETUP
    "sharing_strategy": "grad_sharing",
    "sharing_strategy.num_coms_per_round": 50,
    "sharing_strategy.retrain.num_epochs": 5,
    "sharing_strategy.log_freq": 1,
}

# convert dict to DictConfig
cfg = DictConfig(cfg)
AgentCls = ModelFisherSyncAgent

print(cfg.dataset)

graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg = setup_experiment(
    cfg)


FleetCls = get_fleet(cfg.sharing_strategy, cfg.parallel)

fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                 LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                 train_kwargs=train_cfg, **fleet_additional_cfg)
