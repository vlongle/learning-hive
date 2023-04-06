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

import time
import datetime
import logging
from shell.utils.experiment_utils import setup_experiment
from pprint import pprint
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pprint(cfg)


if __name__ == "__main__":
    main()
