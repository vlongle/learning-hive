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

import time
import datetime
import logging
from shell.utils.experiment_utils import setup_experiment
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start = time.time()

    AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.algo, cfg.parallel)

    graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg = setup_experiment(
        cfg)

    FleetCls = get_fleet(cfg.sharing_strategy, cfg.parallel)

    fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                     LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                     train_kwargs=train_cfg, **fleet_additional_cfg)

    for task_id in range(cfg.dataset.num_tasks):
        fleet.train_and_comm(task_id)

    end = time.time()
    logging.info(f"Run took {datetime.timedelta(seconds=end-start)}")


if __name__ == "__main__":
    main()
