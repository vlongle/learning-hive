'''
File: /helper.py
Project: fleet
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from shell.fleet.fleet import Fleet, ParallelFleet, ParallelAgent, Agent
from shell.fleet.grad.monograd import ModelSyncAgent, ParallelModelSyncAgent
from shell.fleet.grad.fedprox import FedProxAgent, ParallelFedProxAgent
from shell.fleet.grad.gradient_fleet import GradFleet, ParallelGradFleet
from shell.fleet.grad.modgrad import ModGrad, ParallelModGrad
from shell.fleet.data.data_fleet import DataFleet, ParallelDataFleet
from shell.fleet.data.recv import RecvDataAgent, ParallelRecvDataAgent


FLEET_CLS = {
    "no_sharing": {
        True: ParallelFleet,
        False: Fleet,
    },
    # fedavg
    "gradient": {
        True: ParallelGradFleet,
        False: GradFleet,
    },
    "recv_data": {
        True: ParallelDataFleet,
        False: DataFleet,
    },
    # fedprox
    "fedprox": {
        True: ParallelGradFleet,
        False: GradFleet,
    },
    # "debug_joint":{
    #     True: ParallelGradFleet,
    #     False: GradFleet,
    # },
    "debug_joint":{
        True: ParallelFleet,
        False: Fleet,
    },
    # "sender_data": {},
    # "modmod": {},
}


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
    "gradient": {
        "monolithic":
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
    "fedprox": {
        "monolithic": {
            True: ParallelFedProxAgent,
            False: FedProxAgent, 
        },
    },
    "recv_data": {
        "monolithic": {
            True: ParallelRecvDataAgent,
            False: RecvDataAgent,
        },
        "modular": {
            True: ParallelRecvDataAgent,
            False: RecvDataAgent,
        },
    },
    "debug_joint": {
         "monolithic": {
            True: ParallelAgent,
            False: Agent
        },
        "modular": {
            True: ParallelAgent,
            False: Agent
        },
    },
}


def get_fleet(sharing_strategy, parallel=True):
    try:
        return FLEET_CLS[sharing_strategy.name][parallel]
    except KeyError:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} with option parallel={parallel} not implemented")


def get_agent_cls(sharing_strategy, algo, parallel=True):
    try:
        return AGENT_CLS[sharing_strategy.name][algo][parallel]
    except KeyError:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} with option parallel={parallel} not implemented")
