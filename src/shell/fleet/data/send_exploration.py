from abc import abstractmethod
import numpy as np
from typing import (
    Optional,
    Dict,
)


class ExplorationStrategy:
    @abstractmethod
    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray):
        pass

class UniformEpsilonExploration(ExplorationStrategy):
    def __init__(self, num_slates, cfg: Optional[Dict] = {}) -> None:
        self.num_slates = num_slates
        self.epsilon = cfg.get("epsilon", 2.0)
        self.min_epislon = cfg.get("min_epislon", 0.01)
        self.decay_factor = cfg.get("decay_factor", 0.9)
        self.exploit_factor = cfg.get("exploit_factor", 1.0)
        print("epsilon {}, min_epislon {}, decay_factor {}, exploit_factor {}".format(
            self.epsilon, self.min_epislon, self.decay_factor, self.exploit_factor))
        self.step = 0

    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        # HACK: initially we should send a lot of data to get a good estimate of Q
        # if self.step <= 50:
        #     num_slates = self.num_slates * 2
        # else:
        #     num_slates = self.num_slates
        num_slates = min(self.num_slates, observations.shape[0])
        explore_factor = self.epsilon
        weights = explore_factor + Q_values
        # replace nan by 1
        # clamp weights to >= 0
        # HACK: should probably takes exp(weights) and then normalize
        # since clamping to 0 means that negative rewards / Qs will never be picked
        # again, which is appropriate in this oracle preference but might not be true
        # in general (e.g., test improvement)
        # weights = np.maximum(weights, 0)
        weights = (weights / self.epsilon) * self.exploit_factor
        # numerically stable softmax
        weights = np.exp(weights - np.max(weights))
        probs = weights / np.sum(weights)
        # sample without replacement if there's enough non-zero weights for num_slates
        # otherwise, send all non-zero weights
        if np.sum(weights > 0) < num_slates:
            print("sending all non-zero weights")
            action = np.nonzero(weights)[0]
            print()
            # if action is empty, then all weights are 0, so just a random batch
            # (equally likely to be any batch)
            if action.shape[0] == 0:
                action = np.random.choice(
                    np.arange(observations.shape[0]), num_slates)
        else:
            action = np.random.choice(
                observations.shape[0], num_slates, p=probs, replace=False)
        return action

    def update(self, observations: np.ndarray, actions: np.ndarray):
        print("step {} epsilon {}".format(self.step, self.epsilon))
        self.epsilon = max(self.min_epislon,
                           self.epsilon * self.decay_factor)
        self.step += 1


class RandomRouting(ExplorationStrategy):
    def __init__(self, num_tasks, num_cls, num_slates, cfg={}):
        self.num_slates = num_slates

    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        num_slates = min(self.num_slates, observations.shape[0])
        return np.random.choice(
            observations.shape[0], num_slates, replace=False)

    def update(self, observations: np.ndarray, actions: np.ndarray):
        pass


class PureExploitative(ExplorationStrategy):
    def __init__(self, num_tasks, num_cls, num_slates, cfg={}):
        self.num_slates = num_slates

    def get_action(self, observations: np.ndarray, Q_values: np.ndarray):
        num_slates = min(self.num_slates, observations.shape[0])
        return np.argsort(Q_values)[-num_slates:]

    def update(self, observations: np.ndarray, actions: np.ndarray):
        pass

def get_exploration(strategy):
    if strategy == "uniform_epsilon":
        return UniformEpsilonExploration
    elif strategy == "random_routing":
        return RandomRouting
    elif strategy == "pure_exploitative":
        return PureExploitative
    else:
        raise ValueError("Unknown exploration strategy {}".format(strategy))