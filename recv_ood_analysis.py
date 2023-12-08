import omegaconf
from shell.utils.experiment_utils import *
from shell.fleet.utils.fleet_utils import *
from shell.utils.metric import *
import matplotlib.pyplot as plt
from shell.fleet.network import TopologyGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from shell.fleet.fleet import Agent, Fleet
from shell.fleet.data.data_utilize import *
from shell.fleet.data.recv import *

from sklearn.manifold import TSNE
from torchvision.utils import make_grid
from shell.fleet.data.data_utilize import *
import logging
from sklearn.metrics import f1_score
import os
from shell.fleet.data.recv_utils import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description='Run experiment with a specified seed.')
parser.add_argument('--dataset', type=str, default="mnist", choices=[
                    "mnist", "kmnist", "fashionmnist", "cifar100"], help='Dataset for the experiment.')
parser.add_argument('--algo', type=str, default="modular", choices=[
                    "monolithic", "modular"], help='Algorithm for the experiment.')
parser.add_argument('--prefilter_strategy', type=str, default="oracle", choices=[
                    "oracle", "None", "raw_distance"], help='Prefilter strategy for the experiment.')
parser.add_argument('--scorer', type=str, default="cross_entropy", choices=[
                    "cross_entropy", "least_confidence", "margin", "random"], help='Scorer for the experiment.')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG)


def save_debug_data(fleet):
    for agent in fleet.agents:
        agent.save_debug_data()


def load_debug_data(fleet):
    for agent in fleet.agents:
        agent.load_debug_data()


dataset = args.dataset
algo = args.algo
prefilter_strategy = args.prefilter_strategy
scorer = args.scorer

experiment_folder = "experiment_results"
experiment_name = "vanilla_fix_bug_compute_loss_encodev2"

use_contrastive = True
num_trains_per_class = 64
seed = 0
num_tasks = 10
parallel = False
comm_freq = None  # "None" means no communication, doesn't matter for this analysis


save_dir = get_save_dir(experiment_folder, experiment_name,
                        dataset, algo, num_trains_per_class, use_contrastive, seed)
graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg, cfg = get_cfg(
    save_dir)

cfg.sharing_strategy = DictConfig({
    "name": "recv_data",
    "scorer": scorer,
    "num_queries": 5,
    'num_data_neighbors': 5,
    'num_filter_neighbors': 5,
    'num_coms_per_round': 2,
    "query_score_threshold": 0.0,
    "shared_memory_size": 50,
    "comm_freq": comm_freq,
    "prefilter_strategy": prefilter_strategy,
})


if __name__ == "__main__":
    AgentCls = get_agent_cls(cfg.sharing_strategy, cfg.algo, parallel)
    FleetCls = get_fleet(cfg.sharing_strategy, parallel)
    fleet = FleetCls(graph, cfg.seed, datasets, cfg.sharing_strategy, AgentCls, NetCls=NetCls,
                     LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                     train_kwargs=train_cfg, **fleet_additional_cfg)

    ret_confs, ret_id_confs = [], []
    for task_id in tqdm(range(num_tasks)):
        fleet.load_model_from_ckpoint(task_ids=task_id)
        fleet.update_replay_buffers(task_id)
        fleet.communicate(task_id=task_id, start_com_round=0)
        save_debug_data(fleet)

        load_debug_data(fleet)
        compute_recv_fleet_quality(fleet, task_id)

        confs, id_confs = load_recv_fleet_quality(fleet, task_id)
        ret_confs += confs
        ret_id_confs += id_confs

    ret_confs, ret_id_confs = sum(ret_confs), sum(ret_id_confs)
    print(acc(ret_confs), acc(ret_id_confs))
