import numpy as np


def init_Q(env):
    "initialize empty Q table given the environment"
    return np.zeros((env.nS, env.nA))


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
            action = np.random.choice(np.arange(self.Q.shape[1]))
        else:
            action = self.Q[obs].argmax()
        return action


def get_samples_used(episode_lenghts):
    "given the outputted episode_lengths list, returns the number of samples used"
    return sum(episode_lenghts)


def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n


def get_env(name):
    """gets initialize environment with given name by looking at
    constants.NAME2ENV"""
    try:
        return NAME2ENV[name]()
    except KeyError:
        raise KeyError("No such environment defined in constants.py")
