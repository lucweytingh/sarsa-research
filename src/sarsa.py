import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm
import time

import matplotlib.pyplot as plt
import sys


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
    time_start = time.time()
    # Keeps track of useful statistics
    stats = []
    diffs = []
    for i_episode in _tqdm(range(num_episodes)):
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
        diffs.append((current_Q.argmax(1) != Q.argmax(1)).sum())
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    time_end = time.time()
    time_used = time_end - time_start
    return Q, (episode_lengths, episode_returns), diffs
