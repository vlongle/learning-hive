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

    dataset = datasets[0].trainset[0]
    X = dataset.tensors[0]
    print(X)
    agent = fleet.agents[0].agent
    net = agent.net
    import torch
    task_id = 4
    net.add_tmp_module(task_id)
    # net.load_state_dict(torch.load(
    #     "tmp_no_dropout_no_update_modules_checkpt_results/cifar100_modular_numtrain_256_contrastive/cifar100/modular/seed_0/agent_0/task_4/checkpoint.pt")['model_state_dict'])
    net.load_state_dict(torch.load(
        "tmp_no_dropout_no_update_modules_checkpt_results/cifar100_modular_numtrain_256/cifar100/modular/seed_0/agent_0/task_4/checkpoint.pt")['model_state_dict'])
    # net.load_state_dict(torch.load(
    #     "tmp_no_dropout_no_update_modules_checkpt_results/cifar100_modular_numtrain_256_contrastive/cifar100/modular/seed_0/agent_0/task_3/checkpoint.pt")['model_state_dict'])
    print(net)
    print(net.structure[0])
    print(net.structure[1])
    print(net.structure[2])
    print(net.structure[3])
    print(net.structure[4])

    testloaders = {task: torch.utils.data.DataLoader(testset,
                                                     batch_size=128,
                                                     #  batch_size=256,
                                                     shuffle=False,
                                                     num_workers=0,
                                                     #  num_workers=4,
                                                     pin_memory=True,
                                                     ) for task, testset in enumerate(datasets[0].testset[:(task_id+1)])}
    from shell.utils.experiment_utils import eval_net
    print(eval_net(net, testloaders))

    for t in range(task_id):
        trainloader = torch.utils.data.DataLoader(datasets[0].trainset[t],
                                                  batch_size=128,
                                                  #  batch_size=256,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  #  num_workers=4,
                                                  pin_memory=True,
                                                  )
        agent.update_multitask_cost(trainloader, t)
    print(len(agent.replay_buffers[0]))

    # take one update_modules step!
    cur_trainloader = torch.utils.data.DataLoader(datasets[0].trainset[task_id],
                                                  batch_size=64,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  )

    valloader = torch.utils.data.DataLoader(datasets[0].valset[task_id],
                                            batch_size=128,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            )
    # agent.T = 4
    # agent.train(trainloader=cur_trainloader, valloader=valloader,
    #             task_id=task_id, testloaders=testloaders)

    print(agent.net.structure[4])
    agent.update_modules(cur_trainloader, task_id)
    # for i in range(10):
    #     agent.update_modules(cur_trainloader, task_id)
    # print(eval_net(net, {
    #     4: cur_trainloader,
    # }))

    # X, y = cur_trainloader.dataset.tensors
    # agent.compute_loss(X, y, task_id, log=True)

    # This code returns reasonable loss...
    # for X, Y in cur_trainloader:
    #     Y = Y.to(agent.net.device)
    #     X = torch.cat([X[0], X[1]], dim=0)
    #     X = X.to(agent.net.device, non_blocking=True)
    #     agent.compute_loss(X, Y, task_id, log=True)

    print(eval_net(net, testloaders))

    # for task_id in range(cfg.dataset.num_tasks):
    #     fleet.train(task_id)
    #     fleet.communicate(task_id)

    end = time.time()
    logging.info(f"Run took {datetime.timedelta(seconds=end-start)}")


if __name__ == "__main__":
    main()
