'''
File: /helper.py
Project: fleet
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.fleet.fleet import Fleet, ParallelFleet, ParallelAgent, Agent
from shell.fleet.grad.monograd import ModelSyncAgent, ParallelModelSyncAgent
from shell.fleet.grad.gradient_fleet import GradFleet, ParallelGradFleet
from shell.fleet.grad.modgrad import ModGrad, ParallelModGrad


FLEET_CLS = {
    "no_sharing": {
        True: ParallelFleet,
        False: Fleet,
    },
    "gradient": {
        True: ParallelGradFleet,
        False: GradFleet,
    },
}


def get_fleet(sharing_strategy, parallel=True):
    try:
        return FLEET_CLS[sharing_strategy.name][parallel]
    except KeyError:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} with option parallel={parallel} not implemented")


AGENT_CLS = {
    "no_sharing": {
        "monolithic": {
            True: ParallelAgent,
            False: Agent
        },
        "modular": {
            True: ParallelAgent,
            False: Agent
        },
    },
    "gradient":
    {"monolithic":
        {
            True: ParallelModelSyncAgent,
            False: ModelSyncAgent
        },
        "modular":
        {
            True: ParallelModGrad,
            False: ModGrad,
        },
     },
}


def get_agent_cls(sharing_strategy, algo, parallel=True):
    try:
        return AGENT_CLS[sharing_strategy.name][algo][parallel]
    except KeyError:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} with option parallel={parallel} not implemented")
