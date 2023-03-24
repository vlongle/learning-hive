'''
File: /helper.py
Project: fleet
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.fleet.fleet import Fleet, ParallelFleet, ParallelAgent, Agent
from shell.fleet.monograd import ModelSyncAgent, ParallelModelSyncAgent


def get_fleet(parallel=True):
    return ParallelFleet if parallel else Fleet


def get_agent_cls(sharing_strategy, parallel=True):
    if sharing_strategy.name == "no_sharing":
        if parallel:
            return ParallelAgent
        else:
            return Agent
    elif sharing_strategy.name == "gradient":
        if parallel:
            return ParallelModelSyncAgent
        else:
            return ModelSyncAgent
    else:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} not implemented")
