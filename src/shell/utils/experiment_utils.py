'''
File: /experiment_utils.py
Project: utils
Created Date: Tuesday March 21st 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import omegaconf
import subprocess
import torch.nn as nn
import torch
import os
from omegaconf import DictConfig
from shell.datasets.datasets import get_dataset
from shell.utils.utils import seed_everything
from pprint import pprint
from shell.fleet.network import TopologyGenerator
from shell.models.cnn_soft_lifelong_dynamic import CNNSoftLLDynamic
from shell.models.cnn import CNN
from shell.models.mlp import MLP
from shell.models.mlp_soft_lifelong_dynamic import MLPSoftLLDynamic
from shell.learners.er_dynamic import CompositionalDynamicER
from shell.learners.er_nocomponents import NoComponentsER
from shell.datasets.datasets import CombinedDataset
from copy import deepcopy


def process_dataset_cfg(cfg):
    dataset_cfg = dict(cfg.dataset)
    dataset_cfg |= {"num_init_tasks": cfg.num_init_tasks,
                    "use_contrastive": cfg.agent.use_contrastive, }

    dataset_cfg["num_train_per_task"] = dataset_cfg["num_trains_per_class"] * \
        dataset_cfg["num_classes_per_task"]
    del dataset_cfg["num_trains_per_class"]
    dataset_cfg["num_val_per_task"] = dataset_cfg["num_vals_per_class"] * \
        dataset_cfg["num_classes_per_task"]
    del dataset_cfg["num_vals_per_class"]

    return dataset_cfg


def setup_experiment(cfg: DictConfig):
    """
    Seed and return dataset, learner, agent, and net
    for fully reproducible experiments.
    """
    pprint(cfg)
    seed_everything(cfg.seed)
    dataset_cfg = process_dataset_cfg(cfg)

    if dataset_cfg['dataset_name'] == "combined":
        datasets = []
        available_dataset_names = ['mnist', 'kmnist', 'fashionmnist']
        # for each agent, randomly choose a dataset
        for i in range(cfg.num_agents):
            dataset_name = available_dataset_names[i % len(
                available_dataset_names)]
            agent_dataset_cfg = deepcopy(dataset_cfg)
            agent_dataset_cfg['dataset_name'] = dataset_name
            datasets.append(get_dataset(**agent_dataset_cfg))
    else:
        datasets = [get_dataset(**dataset_cfg) for _ in range(cfg.num_agents)]

    net_cfg = dict(cfg.net)
    agent_cfg = dict(cfg.agent)
    train_cfg = dict(cfg.train)

    x = datasets[0].testset[0][0][0]

    i_size = x.shape[1]
    num_classes = datasets[0].num_classes
    print("i_size", i_size)
    print("num_classes", num_classes)

    net_cfg |= {"i_size": i_size,
                "num_classes": num_classes, "num_tasks": cfg.dataset.num_tasks,
                "num_init_tasks": cfg.num_init_tasks,
                "use_contrastive": cfg.agent.use_contrastive, }
    agent_cfg |= {'dataset_name': cfg.dataset.dataset_name}
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
    fleet_additional_cfg = {}

    # REQUIRED_JOINT_TRAINING_STRAT = ["fedprox", "gradient", "debug_joint"]
    REQUIRED_JOINT_TRAINING_STRAT = ["fedprox", "gradient"]

    # if cfg.sharing_strategy.name in REQUIRED_JOINT_TRAINING_STRAT:
    if getattr(cfg.sharing_strategy, 'sync_base', False):
        fake_dataset = get_dataset(**process_dataset_cfg(cfg))
        if dataset_cfg['dataset_name'] == "combined":
            for i in range(fake_dataset.num_tasks):
                chosen_d = datasets[i % len(datasets)]
                task_id = chosen_d.class_sequence[i * chosen_d.num_classes_per_task: (
                    i+1) * chosen_d.num_classes_per_task]
                fake_dataset.testset.append(chosen_d.testset[i])
                fake_dataset.trainset.append(chosen_d.trainset[i])
                fake_dataset.valset.append(chosen_d.valset[i])
                fake_dataset.class_sequence.extend(task_id)
        fleet_additional_cfg['fake_dataset'] = fake_dataset
    return graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg


def load_net(cfg, NetCls, net_cfg, agent_id, task_id, num_added_components=None,
             checkpt_name=None):

    if checkpt_name is None:
        checkpt_name = "checkpoint.pt"

    save_dir = os.path.join(cfg['agent']['save_dir'],
                            f'agent_{agent_id}', f'task_{task_id}')
    if agent_id == 69420:  # HACK for the jointly trained agent
        net_cfg = deepcopy(net_cfg)
        net_cfg['num_tasks'] += net_cfg['num_init_tasks']
        print("net_cfg", net_cfg)

    net = NetCls(**net_cfg)
    # TODO: how to know how many components to add?
    if num_added_components is not None:
        for _ in range(num_added_components):
            # TODO: this is buggy. We need to log the number of new components added
            # over time into a record.csv and use that
            # net.add_tmp_modules(task_id=len(net.components)+1, num_modules=1)
            net.add_tmp_modules(task_id=len(net.components), num_modules=1)

    # print('net', net.components, net.decoder, net.structure)
    checkpoint = torch.load(os.path.join(save_dir, checkpt_name))
    # print('checkpt:', checkpoint['model_state_dict'].keys())
    if "model_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(checkpoint)
    return net


@torch.inference_mode()
def eval_net_task(net, task, testloader):
    a = 0.
    n = len(testloader.dataset)
    for X, Y in testloader:
        if isinstance(X, list):
            X, _ = X
        X = X.to(net.device, non_blocking=True)
        Y = Y.to(net.device, non_blocking=True)
        Y_hat = net(X, task)
        a += (Y_hat.argmax(dim=1) == Y).sum().item()
    return a / n



@torch.inference_mode()
def eval_net(net, testloaders, include_avg=False):
    was_training = net.training
    net.eval()
    test_acc = {}
    for task, loader in testloaders.items():
        test_acc[task] = eval_net_task(net, task, loader)

    if include_avg:
        test_acc['avg'] = sum(test_acc.values()) / len(test_acc)
    if was_training:
        net.train()
    return test_acc



def check_valid_config(cfg):
    # check that 0.8 * datasets.num_trains_per_class is "close" to datasets.num_vals_per_class
    check = 0.8 * cfg["dataset.num_trains_per_class"] - \
        cfg["dataset.num_vals_per_class"]
    # close depends on the magnitude of dataset.num_vals_per_class
    check = check / cfg["dataset.num_vals_per_class"]
    return (abs(check) < 0.1) or (cfg["dataset.num_vals_per_class"] == -1 and cfg["dataset.num_trains_per_class"] == -1)


def get_all_combinations(config, strict=True):
    """
    Config is a dictionary where keys are string,
    and values are either a single value or a list of values.
    Return a list of dictionaries, where in each dictionary,
    a key is a string and a value is a single value by combining
    all possible values from config.
    """
    keys = list(config.keys())
    values = list(config.values())
    combs = []
    for i in range(len(values)):
        if type(values[i]) is not list:
            values[i] = [values[i]]
    for i in range(len(values[0])):
        combs.append({keys[0]: values[0][i]})
    for i in range(1, len(values)):
        new_combs = []
        for j in range(len(combs)):
            for k in range(len(values[i])):
                new_combs.append(combs[j].copy())
                new_combs[-1][keys[i]] = values[i][k]
        combs = new_combs
    if strict:
        # filter out invalid configs
        combs = [c for c in combs if check_valid_config(c)]
    return combs


def get_job_name(dataset, algo, num_trains_per_class, use_contrastive):
    job_name = f"{dataset}_{algo}_numtrain_{num_trains_per_class}"
    job_name += "" if use_contrastive == False else "_contrastive"
    return job_name


def get_save_dir(experiment_folder, experiment_name, dataset, algo, num_trains_per_class, use_contrastive, seed):
    root_save_dir = f"{experiment_folder}/{experiment_name}"
    job_name = get_job_name(
        dataset, algo, num_trains_per_class, use_contrastive)
    return f"{root_save_dir}/{job_name}/{dataset}/{algo}/seed_{seed}"


def get_save_dirv2(root_save_dir, job_name, dataset_name, algo, seed):
    return f"{root_save_dir}/{job_name}/{dataset_name}/{algo}/seed_{seed}"


def get_cfg(save_dir):
    config_path = os.path.join(save_dir, "hydra_out", ".hydra", "config.yaml")
    cfg = omegaconf.OmegaConf.load(config_path)
    graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg = setup_experiment(
        cfg)
    return graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg, fleet_additional_cfg, cfg


def run_experiment(config, strict=True):
    """
    Generate all the combinations from config
    and run them in *sequence* (subprocess.run is blocking).
    """
    script_path = os.path.join("experiments", "run.py")

    print(config)
    combs = get_all_combinations(config, strict=strict)
    print("No. of experiments:", len(combs))

    for cfg in combs:
        dataset = cfg["dataset"]
        algo = cfg["algo"]
        cmd = [
            "python",
            script_path,
        ] + [f"{k}={v}" for k, v in cfg.items()]
        num_trains_per_class = cfg["dataset.num_trains_per_class"]
        job_name = get_job_name(
            dataset, algo, num_trains_per_class, cfg["agent.use_contrastive"])
        cmd += [f"train={algo}", f"job_name={job_name}"]
        # print(" ".join(cmd), "\n\n")
        subprocess.run(cmd)
