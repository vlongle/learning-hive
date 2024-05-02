'''
File: /run.py
Project: experiments
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import hydra
from omegaconf import DictConfig

from shell.fleet.utils.fleet_utils import get_fleet, get_agent_cls
import os

import time
import datetime
import logging
from shell.utils.experiment_utils import setup_experiment, get_save_dirv2
from shell.fleet.combine.combine_modes import *
logging.basicConfig(level=logging.INFO)


def handle_combine_modes(cfg):
    if cfg.sharing_strategy.name != "combine_modes":
        return cfg

    comm_freqs = {}
    num_coms_per_round = {}
    pre_or_post_comm = {}
    for comm in cfg.sharing_strategy.communicator.split(','):
        config = load_comm_config(comm)
        if config['comm_freq'] == "None":
            config['comm_freq'] = None
        comm_freqs[comm] = config['comm_freq']
        num_coms_per_round[comm] = config['num_coms_per_round']
        pre_or_post_comm[comm] = config['pre_or_post_comm']

    cfg.sharing_strategy.comm_freq = comm_freqs
    cfg.sharing_strategy.num_coms_per_round = num_coms_per_round
    cfg.sharing_strategy.pre_or_post_comm = pre_or_post_comm
    return cfg


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start = time.time()

    # save_dir = get_save_dirv2(cfg.root_save_dir, cfg.job_name, cfg.dataset.dataset_name,
    #                           cfg.algo, cfg.seed)
    # if os.path.exists(save_dir) and cfg.overwrite is False:
    #     print(save_dir, "already exists")
    #     return

    AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.algo, cfg.parallel)

    graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg = setup_experiment(
        cfg)

    cfg = handle_combine_modes(cfg)

    # check if cfg.root_save_dir already exists

    FleetCls = get_fleet(cfg.sharing_strategy, cfg.parallel)

    fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                     LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                     train_kwargs=train_cfg, **fleet_additional_cfg)

    for task_id in range(cfg.dataset.num_tasks):
        fleet.train_and_comm(task_id)
        if task_id == 4:
            break

    end = time.time()
    logging.info(f"Run took {datetime.timedelta(seconds=end-start)}")


if __name__ == "__main__":
    main()
