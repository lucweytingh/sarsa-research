import random

import gym
import numpy as np
import fire

from src.experimentresults import ExperimentResults
from src.sarsa import NAME2ALG


def perform_grid_search(
    sarsa_alg, env, nof_alphas=30, random_seeds=[42, 420], n_episodes=1000
):
    """for given sarsa algorithm, environment and parameters return the optimal
    alpha learning rate"""
    print(f"Finding Optimal Alpha for {sarsa_alg}")
    alpha2performance = {}
    for alpha in np.linspace(0.1, 1, nof_alphas):
        perfs = []
        for seed in random_seeds:
            np.random.seed(seed)
            random.seed(seed)
            env.seed(seed)
            (_, (_, episode_returns_sarsa, _), _,) = sarsa_alg(
                env,
                n_episodes,
                alpha=alpha,
            )
            perf = np.mean(episode_returns_sarsa[-100:])
            perfs.append(perf)
            print(seed, alpha, perf)
        alpha2performance[alpha] = np.mean(perfs)
    opt_alpha = max(alpha2performance, key=alpha2performance.get)
    return opt_alpha


def get_alg2metadata(env_name, nof_alphas, seeds, n_episodes):
    alg2metadata = {
        alg: {"nof_alphas": nof_alphas, "seeds": seeds}
        for alg in NAME2ALG.keys()
    }
    env = gym.envs.make(env_name)
    for algname, alg in NAME2ALG.items():
        alg2metadata[algname]["opt_alpha"] = perform_grid_search(
            alg, env, nof_alphas, seeds, n_episodes
        )
    return alg2metadata


def run(env_name, nof_alphas=10, seeds=[42, 420, 4200], n_episodes=1000):
    expresults = ExperimentResults.from_storage()
    if expresults.results_present(env_name, nof_alphas, seeds):
        answer = input(
            "Results for this environment and these parameters have already\
 been computed, continue? (y/n) "
        )
        if not answer.startswith("y"):
            expresults.show_results(env_name)
            return
    alg2metadata = get_alg2metadata(env_name, nof_alphas, seeds, n_episodes)
    expresults[env_name].update(alg2metadata)
    expresults.show_results(env_name)
    expresults.write()


if __name__ == "__main__":
    fire.Fire(run)
