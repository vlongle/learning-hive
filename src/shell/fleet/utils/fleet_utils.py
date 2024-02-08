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
from shell.fleet.grad.gradient_fleet import SyncBaseFleet, ParallelSyncBaseFleet
from shell.fleet.grad.modgrad import ModGrad, ParallelModGrad
from shell.fleet.data.data_fleet import DataFleet, ParallelDataFleet
from shell.fleet.data.recv import RecvDataAgent, ParallelRecvDataAgent
from shell.fleet.mod.modmod import ModModAgent, ParallelModModAgent


BASIC_FLEET_CLS = {
    "sync_base": {
        True: ParallelSyncBaseFleet,
        False: SyncBaseFleet,
    },
    "no_sync_base": {
        True: ParallelFleet,
        False: Fleet,
    },
}

# FLEET_CLS = {
#     "no_sharing": {
#         True: ParallelFleet,
#         False: Fleet,
#     },
#     # fedavg
#     "gradient": {
#         True: ParallelSyncBaseFleet,
#         False: SyncBaseFleet,
#     },
#     "recv_data": {
#         True: ParallelDataFleet,
#         False: DataFleet,
#     },
#     # fedprox
#     "fedprox": {
#         True: ParallelSyncBaseFleet,
#         False: SyncBaseFleet,
#     },
#     # "debug_joint":{
#     #     True: ParallelGradFleet,
#     #     False: GradFleet,
#     # },
#     "debug_joint": {
#         True: ParallelFleet,
#         False: Fleet,
#     },
#     # "sender_data": {},
#     "modmod": {
#         True: ParallelFleet,
#         False: Fleet,
#     },
# }


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
    "modmod": {

        'monolithic': {
            True: ParallelModModAgent,
            False: ModModAgent,
        },

        'modular': {
            True: ParallelModModAgent,
            False: ModModAgent,
        }
    },
}


def get_fleet(sharing_strategy, parallel=True):
    try:
        sync_base = "sync_base" if sharing_strategy.sync_base else "no_sync_base"
        return BASIC_FLEET_CLS[sync_base][parallel]
    except KeyError:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} with option parallel={parallel} not implemented")


def get_agent_cls(sharing_strategy, algo, parallel=True):
    try:
        return AGENT_CLS[sharing_strategy.name][algo][parallel]
    except KeyError:
        raise NotImplementedError(
            f"sharing strategy {sharing_strategy.name} with option parallel={parallel} not implemented.")
