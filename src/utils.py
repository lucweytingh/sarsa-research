from src.constants import NAME2ENV
import numpy as np
from collections import defaultdict


def init_Q(env):
    "initialize empty Q table given the environment"
    if len(env.observation_space.shape) > 1:
        return dictQ(env)
    else:
        return matrixQ(env)


def float_ddict():
    return defaultdict(float)


class dictQ:
    def __init__(self, env):
        self.state2action2Q = defaultdict(float_ddict)
        self.env = env

    def get_best_action(self, state):
        action2Q = self.state2action2Q[state]
        if len(action2Q.keys()) == 0:
            return self.action_space.sample()
        else:
            return max(action2Q, key=action2Q.get)

    def get(self, state, action):
        return self.state2action2Q[state][action]


class matrixQ:
    def __init__(self, env):
        self.Q = np.zeros((env.nS, env.nA))
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
    "given the outputted episode_lengths list, returns the number of samples used"
    return sum(episode_lenghts)


def stopping_criterion_mean_lt(n=5, for_last=100):
    def stopping_criterion(diffs):
        """given the change in return over episodes, return True iff we consider
        the algorithm converged"""
        return len(diffs) > for_last and np.mean(diffs[-for_last:]) < n


stopping_criteria = {
    "mean_lt": stopping_criterion_mean_lt,
    "mean_lt_default": stopping_criterion_mean_lt(),
    "never": lambda *args: False,
}


def resolve_stopping_criterion(stopping_criterion):
    if isinstance(stopping_criterion, str):
        try:
            return stopping_criteria[stopping_criterion]()
        except KeyError:
            raise KeyError("No such stopping_criterion defined.")
    else:
        return stopping_criterion


def get_env(name):
    """gets initialize environment with given name by looking at
    constants.NAME2ENV"""
    try:
        return NAME2ENV[name]()
    except KeyError:
        raise KeyError("No such environment defined in constants.py")
