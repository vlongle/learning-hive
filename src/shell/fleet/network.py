'''
File: /network.py
Project: fleet
Created Date: Monday March 20th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import numpy as np


class TopologyGenerator:
    def __init__(self, num_nodes: int, edge_drop_prob: float = 0.0):
        self.num_nodes = num_nodes
        self.edge_drop_prob = edge_drop_prob

    def generate_fully_connected(self):
        G = nx.complete_graph(self.num_nodes)
        # G = self.__drop_edges(G)
        return G

    def generate_disconnected(self):
        # graph with no edges (i.e., no communication)
        G = nx.empty_graph(self.num_nodes)
        return G

    def generate_tree(self):
        # Generates a random tree topology
        return nx.random_tree(self.num_nodes)

    def generate_server(self):
        # Generates a server-based topology
        G = nx.empty_graph(self.num_nodes)
        for i in range(1, self.num_nodes):
            G.add_edge(0, i)  # Assuming node 0 is the server
        return G

    def generate_self_loop(self):
        # graph with self-loops (i.e., only communicate with self)
        G = nx.empty_graph(self.num_nodes)
        for i in range(self.num_nodes):
            G.add_edge(i, i)
        return G

    def generate_ring(self):
        G = nx.cycle_graph(self.num_nodes)
        # G = self.__drop_edges(G)
        return G

    def generate_random(self, edge_drop_prob=None):
        if edge_drop_prob is None:
            edge_drop_prob = self.edge_drop_prob
        G = nx.erdos_renyi_graph(self.num_nodes, p=1-edge_drop_prob)
        return G

    def generate_connected_random(self, edge_drop_prob=None):
        if edge_drop_prob is None:
            edge_drop_prob = self.edge_drop_prob

        # Step 1: Create a random spanning tree to ensure connectivity
        G = nx.random_tree(self.num_nodes)

        # Step 2: Add additional edges with probability p=1-edge_drop_prob
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):  # avoid duplicate edges and self-loops
                if not G.has_edge(i, j) and random.random() <= (1 - edge_drop_prob):
                    G.add_edge(i, j)

        return G

    def __drop_edges(self, G: nx.Graph):
        edges = list(G.edges())
        for edge in edges:
            if random.random() <= self.edge_drop_prob:
                G.remove_edge(edge[0], edge[1])
        return G

    @staticmethod
    def save_graph(G: nx.Graph, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(G, f)

    @staticmethod
    def load_graph(filename: str):
        with open(filename, "rb") as f:
            G = pickle.load(f)
        return G


def set_color(fleet):
    # Create a mapping from agent id to dataset name
    agent_dataset_mapping = {
        agent.node_id: agent.dataset.name for agent in fleet.agents}
    dataset_colors = {
        'mnist': '#E41A1C',  # Cherry Red
        'kmnist': '#377EB8',  # Sapphire Blue
        'fashionmnist': '#4DAF4A',  # Apple Green
        'cifar100': '#984EA3',  # Amethyst Purple
    }

    # Now, add the dataset name as an attribute to each node in the graph
    for node in fleet.graph.nodes:
        # Assuming node corresponds to agent.id
        if node in agent_dataset_mapping:
            dataset_name = agent_dataset_mapping[node]
            fleet.graph.nodes[node]['dataset'] = dataset_colors.get(
                dataset_name, 'grey')  # Default color


def get_communication(fleet, multiplier=1, exponential=False):
    N = len(fleet.agents)

    # Initialize matrices for summing task_sim values and counting occurrences
    communication_sum = np.zeros((N, N))
    communication_count = np.zeros((N, N))

    for i, agent in enumerate(fleet.agents):
        df = agent.modmod_record.df
        for _, row in df.iterrows():
            j = int(row['neighbor_id'])
            communication_sum[i, j] += row['task_sim']
            communication_count[i, j] += 1

    # Calculate the average only where there's at least one occurrence
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by zero
        communication_avg = np.divide(communication_sum, communication_count, out=np.zeros_like(
            communication_sum), where=communication_count != 0)

    if exponential:
        # Apply exponential function to non-zero averages
        communication_avg[communication_avg != 0] = np.exp(
            communication_avg[communication_avg != 0] * multiplier)

    return communication_avg


def set_weight(fleet, communication):
    G = fleet.graph
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        # Assuming u, v are indices in the communication matrix
        if u < len(communication) and v < len(communication):
            d['weight'] = communication[u, v]
        # else:
        #     d['weight'] = 1  # Default width

    edge_widths = [d['weight'] for _, _, d in G.edges(data=True)]
    return edge_widths


def compute_cluster_communications(fleet, communication):
    # Retrieve node colors (cluster identifications)
    node_colors = {node: data['dataset']
                   for node, data in fleet.graph.nodes(data=True)}

    # Initialize variables to store total communications and counts
    in_cluster_total = 0.0
    out_cluster_total = 0.0
    in_cluster_count = 0
    out_cluster_count = 0

    N = len(fleet.agents)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # Skip self-communications
            # Check if both nodes exist and have a color assigned
            if i in node_colors and j in node_colors:
                if node_colors[i] == node_colors[j]:  # In-cluster
                    in_cluster_total += communication[i, j]
                    in_cluster_count += 1
                else:  # Out-cluster
                    out_cluster_total += communication[i, j]
                    out_cluster_count += 1

    # Compute average communications, handling division by zero
    in_cluster_avg = in_cluster_total / in_cluster_count if in_cluster_count else 0
    out_cluster_avg = out_cluster_total / \
        out_cluster_count if out_cluster_count else 0

    return in_cluster_avg, out_cluster_avg
