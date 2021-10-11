import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm
import time
from src.utils import init_Q, get_env, EpsilonGreedyPolicy, stopping_criterion
import matplotlib.pyplot as plt
import sys
import copy


def sarsa(
    env,
    num_episodes,
    stopping_criterion=stopping_criterion,
    discount_factor=1.0,
    alpha=0.5,
):
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
    Q = init_Q(env)
    policy = EpsilonGreedyPolicy(Q, 0.1)
    time_start = time.time()
    # Keeps track of useful statistics
    stats = []
    diffs = []
    R = 0
    for i_episode in _tqdm(range(num_episodes)):
        policy.Q = Q
        state = env.reset()
        i = 0
        old_R = R
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
        diff = abs(old_R - R)
        diffs.append(diff)
        if stopping_criterion(diffs):
            episode_lengths, episode_returns = zip(*stats)
            time_used = time.time() - time_start
            return Q, (episode_lengths, episode_returns), diffs, time_used

    episode_lengths, episode_returns = zip(*stats)
    time_used = time.time() - time_start
    return Q, (episode_lengths, episode_returns), diffs, time_used


def expected_sarsa(
    env,
    num_episodes,
    stopping_criterion=stopping_criterion,
    discount_factor=1.0,
    alpha=0.5,
):
    """
    expected SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        stopping_criterion: function that takes list of differences and returns
           True if converged
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.

    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    Q = init_Q(env)
    policy = EpsilonGreedyPolicy(Q, 0.1)
    time_start = time.time()
    # Keeps track of useful statistics

    stats = []
    diffs = []
    R = 0
    for i_episode in _tqdm(range(num_episodes)):
        s = env.reset()
        i = 0
        old_R = R
        R = 0
        a = policy.sample_action(s)
        while True:
            s_, r, done, _ = env.step(a)
            a_ = policy.sample_action(s_)
            q_max = max(Q[s_, :])
            non_greedy_action_probability = policy.epsilon / env.nA
            greedy_action_probability = (
                (1 - policy.epsilon)
            ) + non_greedy_action_probability
            expected_q = sum(
                Q[s_][i] * greedy_action_probability
                if Q[s_][i] == q_max
                else Q[s_][i] * non_greedy_action_probability
                for i in range(env.nA)
            )
            Q[s, a] = Q[s, a] + alpha * (
                r + discount_factor * expected_q - Q[s, a]
            )
            s = s_
            a = a_
            R += r
            i += 1
            if done:
                break
        diff = abs(old_R - R)
        diffs.append(diff)
        stats.append((i, R))
        if stopping_criterion(diffs):
            episode_lengths, episode_returns = zip(*stats)
            time_used = time.time() - time_start
            return Q, (episode_lengths, episode_returns), diffs, time_used
    time_used = time.time() - time_start
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns), diffs, time_used


# from windy_gridworld import WindyGridworldEnv

# env = WindyGridworldEnv()


# def running_mean(vals, n=1):
#     cumvals = np.array(vals).cumsum()
#     return (cumvals[n:] - cumvals[:-n]) / n


# Q = np.zeros((env.nS, env.nA))
# policy = EpsilonGreedyPolicy(Q, epsilon=0.1)
# (
#     Q_sarsa,
#     (episode_lengths_sarsa, episode_returns_sarsa),
#     diffs,
# ) = sarsa(env, policy, Q, 1000)
# print(len(episode_lengths_sarsa))
# n = 50
# # We will help you with plotting this time
# plt.clf()
# plt.plot(running_mean(diffs, n))
# plt.title("diffs")
# plt.show()
