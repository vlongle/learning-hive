'''
File: /test_grad.py
Project: experiments
Created Date: Monday March 27th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.fleet.utils.model_sharing_utils import is_in
from shell.utils.utils import seed_everything
from omegaconf import OmegaConf
from hydra import compose, initialize
import ray
from pprint import pprint
from shell.datasets.datasets import get_dataset
from shell.utils.experiment_utils import setup_experiment, process_dataset_cfg, eval_net
import logging
import datetime
import time
import torch
from shell.fleet.utils.fleet_utils import get_fleet, get_agent_cls
from omegaconf import DictConfig
import hydra
import os


logging.basicConfig(level=logging.INFO)

initialize(config_path="conf", job_name="tmp_job")
cfg = compose(config_name="grad")
seed_everything(cfg.seed)

cfg.parallel = True
cfg.train.num_epochs = 50
cfg.train.component_update_freq = 50

AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.algo, cfg.parallel)

graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg = setup_experiment(
    cfg)
FleetCls = get_fleet(cfg.sharing_strategy, cfg.parallel)

fake_dataset = get_dataset(**process_dataset_cfg(cfg))
fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                 LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                 train_kwargs=train_cfg, fake_dataset=fake_dataset)


def diff_models(modelA_statedict, modelB_statedict, keys=None):
    diffs = {}
    # compute the average difference between two models
    for key in modelA_statedict.keys():
        if keys is not None and not is_in(key, keys):
            continue
        if key in modelB_statedict.keys():
            diffs[key] = torch.mean(
                torch.abs(modelA_statedict[key] - modelB_statedict[key])).item()
    return diffs

# diff_models(fleet.agents[0].get_net().state_dict(), fleet.agents[1].get_net().state_dict(),
#             keys=["random_linear_projection", "components"])


# diff_models(ray.get(fleet.agents[0].get_net.remote()).state_dict(),
#             ray.get(fleet.agents[1].get_net.remote()).state_dict(),
#             keys=["random_linear_projection", "components"])


# local train
fleet.train(0)

# fleet.agents[0].net.add_tmp_module(0)
# ray.get(fleet.agents[0].eval.remote(0))

# diff_models(ray.get(fleet.agents[0].get_net.remote()).state_dict(),
#             ray.get(fleet.agents[1].get_net.remote()).state_dict(),
#             keys=["random_linear_projection", "components"])

# fleet

fleet.num_coms_per_round = 1
fleet.communicate(0)
# evaluate the performance again
# ray.get(fleet.agents[0].eval.remote(0))

# diff_models(ray.get(fleet.agents[0].get_net.remote()).state_dict(),
#             ray.get(fleet.agents[1].get_net.remote()).state_dict(),
#             keys=["random_linear_projection", "components"])
