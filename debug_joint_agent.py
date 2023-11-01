# %%
import os
import omegaconf
from shell.utils.experiment_utils import *
from shell.utils.metric import *
import matplotlib.pyplot as plt
from shell.fleet.network import TopologyGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from shell.fleet.utils.fleet_utils import *

# %%
# num_tasks = 5
num_init_tasks = 4
# algo = "monolithic"
algo = "modular"
comm_freq = 10
use_contrastive = False

checkpt_name = "checkpoint_10.pth"
# checkpt_name = None

experiment = f"debug_experiment_results/small_debug_joint_agent_use_reg_fleet_comm_freq_{comm_freq}_use_contrastive_{use_contrastive}"

# %%
def get_cfg(save_root_dir = "experiment_results/toy_fedprox",
    dataset = "mnist",
    algo = "monolithic",
    num_train = 64,
    seed = 0,
    use_contrastive = True,):
    job_name = f"{dataset}_{algo}_numtrain_{num_train}"
    if use_contrastive:
        job_name += "_contrastive"
    experiment = os.path.join(save_root_dir, job_name, dataset,algo, f"seed_{seed}")
    config_path = os.path.join(experiment, "hydra_out", ".hydra", "config.yaml")
    # read the config file
    cfg = omegaconf.OmegaConf.load(config_path)
    graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg,fleet_additional_cfg = setup_experiment(cfg)
    # net_cfg['num_tasks'] = num_tasks - num_init_tasks 
    return graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg,fleet_additional_cfg, cfg


# %%
graph, datasets, NetCls, LearnerCls, net_cfg, agent_cfg, train_cfg,fleet_additional_cfg, cfg = get_cfg(experiment,
                                                                                                       algo=algo,
                                                                                                       use_contrastive=use_contrastive)

# %%
agent_id = 0
task_id = 4
num_added_components = 1
# num_added_components = None


net = load_net(cfg, NetCls, net_cfg, agent_id=agent_id, task_id=task_id, num_added_components=num_added_components,
checkpt_name=checkpt_name)

if agent_id == 69420:
    dataset = fleet_additional_cfg['fake_dataset']
else:
    dataset = datasets[agent_id]
testloaders = {task: torch.utils.data.DataLoader(testset,
                                                         batch_size=256,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=True,
                                                         ) for task, testset in enumerate(dataset.testset[:(task_id+1)])}

print()
print(eval_net(net, testloaders))
print('\n\n')
print("random proj:", net.random_linear_projection.weight)
print("components:", net.components)
print('\n\n')
if algo == "modular":
    print("structure", net.structure)
    for t in range(task_id+1):
        print(net.structure[t])

print("\n weight of the last component")
print(net.components[-1].weight)
# %%


# %%



