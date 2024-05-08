'''
File: /combine_modes_utils.py
Project: combine
Created Date: Tuesday April 23rd 2024
Author: Long Le (vlongle@seas.upenn.edu)
Copyright (c) 2024 Long Le
'''


from omegaconf import OmegaConf
from shell.fleet.fleet import Fleet, ParallelFleet, ParallelAgent, Agent
from shell.fleet.grad.monograd import ModelSyncAgent, ParallelModelSyncAgent
# from shell.fleet.grad.fedprox import FedProxAgent, ParallelFedProxAgent, FedProxModAgent, ParallelFedProxModAgent
# from shell.fleet.grad.fedcurv import FedCurvAgent, ParallelFedCurvAgent, FedCurvModAgent, ParallelFedCurvModAgent
from shell.fleet.grad.gradient_fleet import SyncBaseFleet, ParallelSyncBaseFleet
from shell.fleet.grad.modgrad import ModGrad, ParallelModGrad
from shell.fleet.data.data_fleet import DataFleet, ParallelDataFleet
from shell.fleet.data.recv import RecvDataAgent, ParallelRecvDataAgent
from shell.fleet.mod.modmod import ModModAgent, ParallelModModAgent
from shell.fleet.data.heuristic import HeuristicDataAgent, ParallelHeuristicDataAgent


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
    # "fedprox": {
    #     "monolithic": {
    #         True: ParallelFedProxAgent,
    #         False: FedProxAgent,
    #     },
    #     "modular": {
    #         True: ParallelFedProxModAgent,
    #         False: FedProxModAgent,
    #     },
    # },
    # "fedcurv": {
    #     "monolithic": {
    #         True: ParallelFedCurvAgent,
    #         False: FedCurvAgent,
    #     },
    #     "modular": {
    #         True: ParallelFedCurvModAgent,
    #         False: FedCurvModAgent,
    #     },
    # },
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
    "heuristic_data": {
        'monolithic': {
            True: ParallelHeuristicDataAgent,
            False: HeuristicDataAgent,
        },
        'modular': {
            True: ParallelHeuristicDataAgent,
            False: HeuristicDataAgent,
        }
    },
}


def load_comm_config(comm_strategy):
    config_path = f"experiments/conf/sharing_strategy/{comm_strategy}.yaml"
    config = OmegaConf.load(config_path)
    return config
