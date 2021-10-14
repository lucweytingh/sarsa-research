import random

import gym
import numpy as np
import fire

from src.experimentresults import ExperimentResults
from src.sarsa import NAME2ALG


def perform_grid_search(sarsa_alg, env, nof_alphas=20, random_seeds=[42, 420]):
    """for given sarsa algorithm, environment and parameters return the optimal
    alpha learning rate"""
    alpha2performance = {}
    for alpha in np.linspace(0.1, 1, nof_alphas):
        perfs = []
        for seed in random_seeds:
            np.random.seed(seed)
            random.seed(seed)
            env.seed(seed)
            (_, (_, episode_returns_sarsa, _), _,) = sarsa_alg(
                env,
                2000,
                alpha=alpha,
            )
            perf = np.mean(episode_returns_sarsa[-100:])
            perfs.append(perf)
            print(seed, alpha, perf)
        alpha2performance[alpha] = np.mean(perfs)
    opt_alpha = max(alpha2performance, key=alpha2performance.get)
    return opt_alpha


def get_alg2metadata(env_name, nof_alphas, seeds):
    alg2metadata = {
        alg: {"nof_alphas": nof_alphas, "seeds": seeds}
        for alg in NAME2ALG.keys()
    }
    env = gym.envs.make(env_name)
    for algname, alg in NAME2ALG.items():
        alg2metadata[algname]["opt_alpha"] = perform_grid_search(
            alg, env, nof_alphas, seeds
        )
    return alg2metadata


def get_alg2alpha(env_name, seeds):
    "returns an alg2alpha for the provided environment name"
    alg2alpha = dict()

    for alg_name, alg in NAME2ALG.items():
        print(alg_name)
        alg2alpha[alg_name] = perform_grid_search(
            alg, env, nof_alphas=3, random_seeds=seeds
        )
    return alg2alpha


def run(env_name, nof_alphas=10, seeds=[42, 420, 4200]):
    expresults = ExperimentResults.from_storage()
    if expresults.results_present(env_name, nof_alphas, seeds):
        answer = input(
            "Results for this environment and these parameters have already been computed, continue? (y/n) "
        )
        if not answer.startswith("y"):
            expresults.show_results(env_name)
            return
    alg2metadata = get_alg2metadata(env_name, nof_alphas, seeds)
    expresults[env_name].update(alg2metadata)
    expresults.show_results(env_name)
    expresults.write()


if __name__ == "__main__":
    fire.Fire(run)
