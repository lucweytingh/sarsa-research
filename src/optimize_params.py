from sarsa import expected_sarsa, sarsa, NAME2ALG
from envs.windy_gridworld import WindyGridworldEnv
from utils import EpsilonGreedyPolicy, init_Q
from constants import ENV_NAMES
from collections import defaultdict

import gym
import numpy as np
import json


def perform_grid_search(sarsa_alg, env, repeats=5, alphas=20):
    alpha2performance = {}
    for alpha in np.linspace(0.1, 1, alphas):
        perfs = []
        for _ in range(repeats):
            Q = init_Q(env)
            policy = EpsilonGreedyPolicy(Q, epsilon=0.1)
            (
                Q_sarsa,
                (episode_lengths_sarsa, episode_returns_sarsa, _),
                diffs,
            ) = sarsa_alg(
                env,
                2000,
                alpha=alpha,
            )
            perf = np.mean(episode_returns_sarsa[-100:])
            perfs.append(perf)
        alpha2performance[alpha] = np.mean(perfs)
    opt_alpha = max(alpha2performance, key=alpha2performance.get)
    return opt_alpha


def update_json(env2alg2alpha, path="../env2alg2alpha.json"):
    with open(path, "w") as f:
        json.dump(env2alg2alpha, f)


if __name__ == "__main__":
    env2alg2alpha = defaultdict(dict)
    for env_name in ENV_NAMES:
        env = gym.make(env_name)
        breakpoint()
        for alg_name, alg in NAME2ALG.items():
            print(f"{env_name},{alg_name}")
            env2alg2alpha[env_name][alg_name] = perform_grid_search(
                alg, env, alphas=1, repeats=1
            )
            update_json(env2alg2alpha)
