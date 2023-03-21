'''
File: /run.py
Project: experiments
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import hydra
from omegaconf import DictConfig
from shell.datasets.datasets import get_dataset
from shell.utils.utils import seed_everything
from shell.fleet.fleet import Fleet, VanillaAgent
from shell.models.cnn_soft_lifelong_dynamic import CNNSoftLLDynamic
from shell.models.cnn import CNN
from shell.models.mlp import MLP
from shell.models.mlp_soft_lifelong_dynamic import MLPSoftLLDynamic
from shell.learners.er_dynamic import CompositionalDynamicER
from shell.learners.er_nocomponents import NoComponentsER
from shell.fleet.network import TopologyGenerator
from pprint import pprint
import time
import datetime
import logging
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    start = time.time()
    pprint(cfg)
    seed_everything(cfg.seed)
    dataset_cfg = dict(cfg.dataset)
    dataset_cfg["num_train_per_task"] = dataset_cfg["num_trains_per_class"] * \
        dataset_cfg["num_classes_per_task"]
    del dataset_cfg["num_trains_per_class"]
    dataset_cfg["num_val_per_task"] = dataset_cfg["num_vals_per_class"] * \
        dataset_cfg["num_classes_per_task"]
    del dataset_cfg["num_vals_per_class"]
    datasets = [get_dataset(**dataset_cfg) for _ in range(cfg.num_agents)]
    net_cfg = dict(cfg.net)
    agent_cfg = dict(cfg.agent)
    train_cfg = dict(cfg.train)

    x = datasets[0].trainset[0][0][0]

    i_size = x.shape[1]
    num_classes = datasets[0].num_classes

    net_cfg |= {"i_size": i_size,
                "num_classes": num_classes, "num_tasks": cfg.dataset.num_tasks}
    print("net_cfg", net_cfg)
    tg = TopologyGenerator(num_nodes=cfg.num_agents)
    graph = tg.generate_random()

    if cfg.algo == "modular":
        if cfg.net.name == "mlp":
            NetCls = MLPSoftLLDynamic
        elif cfg.net.name == "cnn":
            NetCls = CNNSoftLLDynamic
    elif cfg.algo == "monolithic":
        if cfg.net.name == "mlp":
            NetCls = MLP
        elif cfg.net.name == "cnn":
            NetCls = CNN
    else:
        raise NotImplementedError

    if cfg.algo == "modular":
        net_cfg |= {"num_tasks": cfg.dataset.num_tasks, }

    del net_cfg["name"]

    LearnerCls = CompositionalDynamicER if cfg.algo == "modular" else NoComponentsER
    print(LearnerCls)

    print("net_cfg")
    pprint(net_cfg)

    # if cfg.sharing_strategy.name == "data":
    #     num_coms_per_round = 2
    #     AgentCls = ReceiverFirstDataAgent
    # else:
    #     num_coms_per_round = 0
    #     AgentCls = VanillaAgent

    num_coms_per_round = 0
    AgentCls = VanillaAgent

    fleet = Fleet(graph, cfg.seed, datasets, num_coms_per_round, AgentCls, NetCls=NetCls,
                  LearnerCls=LearnerCls, net_kwargs=net_cfg, agent_kwargs=agent_cfg,
                  train_kwargs=train_cfg)

    for task_id in range(cfg.dataset.num_tasks):
        fleet.train(task_id)
        fleet.communicate(task_id)

    # net = NetCls(**net_cfg)
    # agent_cfg.pop("batch_size", None)
    # agent = LearnerCls(net, **agent_cfg)
    # dataset = datasets[0]
    # import torch
    # for task_id in range(cfg.dataset.num_tasks):
    #     trainloader = (
    #         torch.utils.data.DataLoader(dataset.trainset[task_id],
    #                                     batch_size=64,
    #                                     shuffle=True,
    #                                     num_workers=0,
    #                                     pin_memory=True,
    #                                     ))
    #     testloaders = {task: torch.utils.data.DataLoader(testset,
    #                                                      batch_size=128,
    #                                                      shuffle=False,
    #                                                      num_workers=0,
    #                                                      pin_memory=True,
    #                                                      ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}
    #     valloader = torch.utils.data.DataLoader(dataset.valset[task_id],
    #                                             batch_size=128,
    #                                             shuffle=False,
    #                                             num_workers=0,
    #                                             pin_memory=True,
    #                                             )
    #     agent.train(trainloader, task_id, testloaders=testloaders,
    #                 valloader=valloader, **train_cfg)

    end = time.time()
    logging.info(f"Run took {datetime.timedelta(seconds=end-start)}")


if __name__ == "__main__":
    main()
