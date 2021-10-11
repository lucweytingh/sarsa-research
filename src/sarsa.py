import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

import matplotlib.pyplot as plt
import sys


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """

    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.Q.shape[1]))
        else:
            action = self.Q[obs].argmax()
        return action


def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.

    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """

    # Keeps track of useful statistics
    stats = []
    for i_episode in tqdm(range(num_episodes)):
        policy.Q = Q
        state = env.reset()
        i = 0
        R = 0
        action = policy.sample_action(state)
        while True:
            (new_state, reward, done, _) = env.step(action)
            policy.Q = Q
            new_action = policy.sample_action(new_state)
            R += reward * (discount_factor ** i)
            Q[state, action] = Q[state, action] + alpha * (
                reward
                + discount_factor * policy.Q[new_state, new_action]
                - Q[state, action]
            )
            state = new_state
            action = new_action
            i += 1
            if done:
                break
        stats.append((i, R))

    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)


def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n
