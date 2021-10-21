from plot import plot_results
from src.experimentresults import ExperimentResults
from src.sarsa import NAME2ALG

import fire
import optimize_params
import gym
import random

import numpy as np
import matplotlib.pyplot as plt


def run(env_name, n_episodes=3000, n_optimize=5, n_runs=50):
    optim_seeds = np.arange(1, n_optimize + 1).tolist()
    run_seeds = (np.arange(1, n_runs + 1) * 1000).tolist()
    optimize_params.run(env_name, seeds=optim_seeds, n_episodes=n_episodes)
    exp2results = ExperimentResults.from_storage()
    env = gym.envs.make(env_name)
    env.seed(0)
    alg2results = {}
    for name, alg in NAME2ALG.items():
        alpha = exp2results[env_name][name]["opt_alpha"]
        alg2results[name] = get_results(
            alg,
            env,
            alpha,
            n_runs=n_runs,
            seeds=run_seeds,
            n_episodes=n_episodes,
        )
    plt.close("all")
    plot_results(
        alg2results,
        env_name,
        fname=env_name,
        running_mean_n=5000,
    )


def get_results(sarsa_fn, env, alpha, n_runs=3, seeds=None, n_episodes=1000):
    if not seeds:
        seeds = np.arange(n_runs) * 10
    returns = np.zeros((n_episodes, n_runs))
    lengths = np.zeros((n_episodes, n_runs))
    times = np.zeros((n_episodes, n_runs))
    for r in range(n_runs):
        seed = int(seeds[r] * 10)
        random.seed(seed)
        np.random.seed(seed)
        env.reset()
        env.seed(seed)
        env.action_space.seed(seed)
        (
            _,
            (episode_lengths, episode_returns, episode_times),
            _,
        ) = sarsa_fn(env, n_episodes, alpha=alpha)
        returns[:, r] = episode_returns
        lengths[:, r] = episode_lengths
        times[:, r] = episode_times
    return {
        "episode_returns": returns,
        "xvar2results": {"nof updates": lengths, "time": times},
    }


if __name__ == "__main__":
    fire.Fire(run)
