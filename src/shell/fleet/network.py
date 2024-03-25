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
        G = self.__drop_edges(G)
        return G

    def generate_disconnected(self):
        # graph with no edges (i.e., no communication)
        G = nx.empty_graph(self.num_nodes)
        return G

    def generate_self_loop(self):
        # graph with self-loops (i.e., only communicate with self)
        G = nx.empty_graph(self.num_nodes)
        for i in range(self.num_nodes):
            G.add_edge(i, i)
        return G

    def generate_ring(self):
        G = nx.cycle_graph(self.num_nodes)
        G = self.__drop_edges(G)
        return G

    def generate_random(self):
        G = nx.erdos_renyi_graph(self.num_nodes, p=1-self.edge_drop_prob)
        return G

    def generate_connected_random(self):
        # Step 1: Create a random spanning tree to ensure connectivity
        G = nx.random_tree(self.num_nodes)

        # Step 2: Add additional edges with probability p=1-edge_drop_prob
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):  # avoid duplicate edges and self-loops
                if not G.has_edge(i, j) and random.random() <= (1 - self.edge_drop_prob):
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

    @staticmethod
    def plot_graph(G: nx.Graph,
                   node_color="#1f78b4", edge_color="#bfbfbf",
                   node_size=500, font_size=16,
                   font_family="sans-serif",
                   layout="spring", draw_labels=False,
                   node_color_attr=None,  # Existing parameter for node color attribute
                   save_path=None,
                   edge_widths=1):
        plt.clf()
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        # Determine node colors based on the attribute
        if node_color_attr and nx.get_node_attributes(G, node_color_attr):
            node_colors = [G.nodes[n].get(
                node_color_attr, node_color) for n in G.nodes()]
        else:
            node_colors = node_color

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_size)



        nx.draw_networkx_edges(
            G, pos, edge_color=edge_color, width=edge_widths)

        if draw_labels:
            nx.draw_networkx_labels(
                G, pos, font_size=font_size, font_family=font_family)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
