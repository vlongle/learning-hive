'''
File: /test_eff_mono_grad.py
Project: experiments
Created Date: Monday March 27th 2023
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
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
from shell.utils.utils import seed_everything
from shell.fleet.utils.model_sharing_utils import is_in
logging.basicConfig(level=logging.INFO)

# config_path = os.path.join("conf", "grad.yaml")
# # read the config file
# cfg = omegaconf.OmegaConf.load(config_path)
initialize(config_path="conf", job_name="tmp_job")
cfg = compose(config_name="grad")
seed_everything(cfg.seed)
# override
cfg.parallel = True
# sanity check, never update_modules
# cfg.train.component_update_freq = 1000

AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.parallel)

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


print("BEFORE LOCAL")
# print(diff_models(fleet.agents[0].get_net().state_dict(), fleet.agents[1].get_net().state_dict(),
#                   keys=["random_linear_projection", "components"]))

print(diff_models(ray.get(fleet.agents[0].get_net.remote()).state_dict(),
                  ray.get(fleet.agents[1].get_net.remote()).state_dict(),
                  keys=["random_linear_projection", "components"]))
print("\n\n")


# local train
fleet.train(0)

print("\n\n AFTER LOCAL")

# print(diff_models(fleet.agents[0].get_net().state_dict(), fleet.agents[1].get_net().state_dict(),
#                   keys=["random_linear_projection", "components"]))

print(diff_models(ray.get(fleet.agents[0].get_net.remote()).state_dict(),
                  ray.get(fleet.agents[1].get_net.remote()).state_dict(),
                  keys=["random_linear_projection", "components"]))
