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


class TopologyGenerator:
    def __init__(self, num_nodes: int, edge_drop_prob: float = 0.0):
        self.num_nodes = num_nodes
        self.edge_drop_prob = edge_drop_prob

    def generate_fully_connected(self):
        G = nx.complete_graph(self.num_nodes)
        G = self.__drop_edges(G)
        return G

    def generate_ring(self):
        G = nx.cycle_graph(self.num_nodes)
        G = self.__drop_edges(G)
        return G

    def generate_random(self):
        G = nx.erdos_renyi_graph(self.num_nodes, p=1-self.edge_drop_prob)
        return G

    def __drop_edges(self, G: nx.Graph):
        edges = list(G.edges())
        for edge in edges:
            if random.random() <= self.edge_drop_prob:
                G.remove_edge(edge[0], edge[1])
        return G

    @classmethod
    def save_graph(G: nx.Graph, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(G, f)

    @classmethod
    def load_graph(filename: str):
        with open(filename, "rb") as f:
            G = pickle.load(f)
        return G

    @classmethod
    def plot_graph(G: nx.Graph,
                   node_color="#1f78b4", edge_color="#bfbfbf",
                   node_size=500, font_size=16,
                   font_family="sans-serif", edge_width=1,
                   layout="spring", draw_labels=True):
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        nx.draw_networkx_nodes(
            G, pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width)
        if draw_labels:
            nx.draw_networkx_labels(
                G, pos, font_size=font_size, font_family=font_family)
        plt.axis("off")
        plt.show()
