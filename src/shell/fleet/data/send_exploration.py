import random
from typing import Dict, Optional, List

import numpy as np
class ExplorationStrategy:
    def get_action(self, scores, neighbor_id):
        raise NotImplementedError
        
    def update(self, neighbor_id):
        pass

class EpsilonGreedyWholeBatch(ExplorationStrategy):
    def __init__(self, neighbors, sharing_strategy=None):
        self.min_epsilon = sharing_strategy.get('min_epsilon', 0.1) if sharing_strategy else 0.1
        self.eps_decay_rate = sharing_strategy.get('eps_decay_rate', 0.99) if sharing_strategy else 0.99
        self.init_epsilon = sharing_strategy.get('init_epsilon', 1.0) if sharing_strategy else 1.0
        self.epsilons = {neighbor.node_id: self.init_epsilon for neighbor in neighbors}

    def get_action(self, scores, neighbor_id, num_points=1):
        epsilon = self.epsilons.get(neighbor_id, 1.0)
        scores = np.array(scores)
        num_available_points = len(scores)

        if random.random() < epsilon:
            # Exploration: Randomly choose unique indices
            selected_indices = np.random.choice(num_available_points, size=min(num_points, num_available_points), replace=False)
        else:
            # Exploitation: Choose the indices of the top scores
            selected_indices = np.argpartition(-scores, min(num_points, num_available_points))[:min(num_points, num_available_points)]
        
        return list(selected_indices)
        
    def update(self, neighbor_id):
        # Decaying the epsilon value for the provided neighbor_id
        old_epsilon = self.epsilons.get(neighbor_id, 1.0)
        new_epsilon = max(self.min_epsilon, old_epsilon * self.eps_decay_rate)
        self.epsilons[neighbor_id] = new_epsilon


class EpsilonGreedy(EpsilonGreedyWholeBatch):
    def get_action(self, scores, neighbor_id, num_points=1):
        epsilon = self.epsilons.get(neighbor_id, 1.0)
        
        scores = np.array(scores)
        explore_count = np.random.binomial(n=num_points, p=epsilon)  
        
        # Get explore_indices, selected randomly from the range of scores
        explore_indices = np.random.choice(len(scores), explore_count, replace=False)
        
        # Remaining indices are the indices that are left after removing explore_indices
        remaining_indices = np.setdiff1d(np.arange(len(scores)), explore_indices)
        exploit_count = num_points - explore_count
        
        # For exploit, we select the indices of the highest scores among the remaining indices
        if exploit_count > 0 and len(remaining_indices) > 0:
            sorted_indices = np.argsort(scores[remaining_indices])[::-1]  # Sort in descending order
            exploit_indices = remaining_indices[sorted_indices[:exploit_count]]
        else:
            exploit_indices = np.array([], dtype=int)  # Empty array if there is no index to exploit
        
        

        selected_indices = np.concatenate((explore_indices, exploit_indices))
        
        return selected_indices.tolist()


class PureExploitation(ExplorationStrategy):
    def get_action(self, scores, neighbor_id, num_points=1):
        actions = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:num_points]
        return actions


class RandomRouting(ExplorationStrategy):
    def get_action(self, scores, neighbor_id, num_points=1):
        actions = random.sample(range(len(scores)), min(num_points, len(scores)))
        return actions


def get_exploration(strategy_name):
    if strategy_name == 'epsilon_greedy':
        return EpsilonGreedy
    elif strategy_name == 'epsilon_greedy_whole_batch':
        return EpsilonGreedyWholeBatch
    elif strategy_name == 'pure_exploitation':
        return PureExploitation
    elif strategy_name == 'random_routing':
        return RandomRouting
    else:
        raise ValueError(f"Invalid exploration strategy {strategy_name}")