'''
File: /gradient_fleet.py
Project: fleet
Created Date: Thursday March 23rd 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from copy import deepcopy
import logging
import ray
from shell.fleet.fleet import Fleet, ParallelFleet
import networkx as nx
from shell.fleet.utils.model_sharing_utils import exclude_model


class GradFleet(Fleet):
    """
    TODO: rename GradFleet to something else.
    This class is used to train agents on the same initial tasks, and then proceed with individual local training.
    This is used for both sharing weights (grad) and modules (mod)
    """
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs,
                 fake_dataset):
        # NOTE: We create a fake agent to jointly train all agents in the same initial tasks. Then,
        # we pretend that all agents have no initial tasks, and proceed with individual local training

        self.fake_agent = AgentCls(69420, seed, fake_dataset, NetCls, LearnerCls,
                                   deepcopy(net_kwargs), deepcopy(
                                       agent_kwargs),
                                   deepcopy(train_kwargs), deepcopy(sharing_strategy))
        self.joint_training()
        # replace net_kwargs["num_init_tasks"] with -1 as we will do joint training on the init tasks.
        net_kwargs["num_tasks"] -= net_kwargs["num_init_tasks"]
        net_kwargs["num_init_tasks"] = -1
        net_kwargs["init_ordering_mode"] = "uniform"
        super().__init__(graph, seed, datasets, sharing_strategy, AgentCls,
                         NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs)

        # all agents should replace their models with the fake_agent's model
        excluded_params = set(["decoder", "structure", "projector", "random_linear_projection"])
        model = exclude_model(
            self.fake_agent.net.state_dict(), excluded_params)

        for agent in self.agents:
            agent.replace_model(model, strict=False)

    def joint_training(self):
        self.fake_agent.agent.fl_strategy = None # NOTE: important to set this to None, otherwise, 
        # the fake_agent will have stuff like L2 penalty as in fedprox
        for task_id in range(self.fake_agent.net.num_init_tasks): 
            logging.info(f"Joint training on task {task_id} ...")
            self.fake_agent.train(task_id)
        logging.info("DONE TRAINING THE JOINT AGENT...")



class ParallelGradFleet(ParallelFleet, GradFleet):
    def __init__(self, graph: nx.Graph, seed, datasets, sharing_strategy, AgentCls, NetCls, LearnerCls, net_kwargs, agent_kwargs,
                 train_kwargs, fake_dataset):
        self.num_init_tasks = net_kwargs["num_init_tasks"]

        self.fake_agent = AgentCls.options(num_gpus=1).remote(69420, seed, fake_dataset, NetCls,
                                                              LearnerCls, deepcopy(
                                                                  net_kwargs), deepcopy(agent_kwargs),
                                                              deepcopy(train_kwargs), deepcopy(sharing_strategy))

        self.fake_agent.set_fl_strategy.remote(None) # NOTE: important to set this to None, otherwise, 
        # the fake_agent will have stuff like L2 penalty as in fedprox

        # need to do the joint training right here, and free the gpu so that other agents can later use them
        for task_id in range(self.num_init_tasks):
            logging.info(f"Joint training on task {task_id} ...")
            ray.get(self.fake_agent.train.remote(task_id))
        
        excluded_params = set(["decoder", "structure", "projector", "random_linear_projection"])
        # store the fake_agent's model
        self.fake_model = exclude_model(ray.get(self.fake_agent.get_model.remote()),
                                        excluded_params)
        # delete the fake_agent
        ray.kill(self.fake_agent)
        del self.fake_agent

        logging.info("DONE TRAINING THE JOINT AGENT...")

        # replace net_kwargs["num_init_tasks"] with -1 as we will do joint training on the init tasks.
        net_kwargs["num_init_tasks"] = -1
        net_kwargs["num_tasks"] -= self.num_init_tasks
        net_kwargs["init_ordering_mode"] = "uniform"
        super().__init__(graph, seed, datasets, sharing_strategy, AgentCls,
                         NetCls, LearnerCls, net_kwargs, agent_kwargs, train_kwargs)
        # now that all agents are created, we can replace their models with the fake_agent's model
        for agent in self.agents:
            agent.replace_model.remote(self.fake_model, strict=False)
        
        logging.info("READY TO TRAIN INDIVIDUAL AGENTS...")
