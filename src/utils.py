import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import gym

from src.constants import ENV_NAMES


class dictQ:
    def __init__(self, env):
        self.state2action2Q = defaultdict(dict)
        self.env = env
        self.action_space = env.action_space

    def get_best_action(self, state):
        action2Q = self.state2action2Q[state]
        if len(action2Q.keys()) == 0:
            return self.action_space.sample()
        else:
            return max(action2Q.keys(), key=action2Q.get)

    def get(self, state, action):
        return self.state2action2Q[state].get(action, 0.0)

    def set(self, state, action, value):
        self.state2action2Q[state][action] = value


def init_Q(env):
    "initialize empty Q table given the environment"
    return dictQ(env)


class matrixQ:
    def __init__(self, env):
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.env = env

    def get_best_action(self, state):
        return self.Q[state].argmax()

    def get(self, state, action):
        return self.Q[state, action]

    def set(self, state, action, value):
        self.Q[state, action] = value


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """

    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from
        this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        if np.random.random() < self.epsilon:
            action = self.Q.env.action_space.sample()
        else:
            action = self.Q.get_best_action(obs)
        return action


def get_samples_used(episode_lenghts):
    """given the outputted episode_lengths list, returns the number of samples
    used"""
    return sum(episode_lenghts)


def get_nof_actions(env):
    if type(env.action_space) == gym.spaces.tuple_space.Tuple:
        return np.prod([act_space.n for act_space in env.action_space.spaces])
    return env.action_space.n


def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n


def load(fpath):
    if str(fpath).endswith(".json"):
        with open(fpath, "r") as f:
            out = json.loads(f.read())
    return out


def write(fpath, to_write):
    if str(fpath).endswith(".json"):
        with open(fpath, "w") as f:
            f.write(json.dumps(to_write))


def get_env(env_name):
    gym.env
