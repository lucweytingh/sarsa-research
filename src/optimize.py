from sarsa import expected_sarsa, sarsa
from envs.windy_gridworld import WindyGridworldEnv
from utils import EpsilonGreedyPolicy, stopping_criterion, init_Q

import numpy as np


def perform_grid_search(sarsa_alg, env):
    alpha2performance = {}
    for alpha in np.linspace(0.1, 1, 20):
        perfs = []
        for _ in range(10):
            Q = init_Q(env)
            policy = EpsilonGreedyPolicy(Q, epsilon=0.1)
            (
                Q_sarsa,
                (episode_lengths_sarsa, episode_returns_sarsa),
                diffs,
            ) = sarsa_alg(
                env,
                policy,
                Q,
                2000,
                alpha=alpha,
                stopping_criterion=stopping_criterion,
            )
            perf = np.mean(episode_returns_sarsa[-100:])
            perfs.append(perf)
        alpha2performance[alpha] = np.mean(perfs)
    opt_alpha = max(alpha2performance, key=alpha2performance.get)
    return opt_alpha


env = WindyGridworldEnv()
print(perform_grid_search(sarsa, env))
print(perform_grid_search(sarsa, env))
print(perform_grid_search(sarsa, env))
print(perform_grid_search(sarsa, env))
