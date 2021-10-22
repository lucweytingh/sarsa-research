import matplotlib.pyplot as plt
import math
import numpy as np
from src.utils import running_mean


def join_with_final_and(ary):
    if len(ary) == 2:
        return " and ".join(ary)
    all_but_last = ", ".join(ary[:-1])
    last = ary[-1]
    return ", and ".join([all_but_last, last])


ALG2COLOR = {"sarsa": "dodgerblue", "expected_sarsa": "darkorange"}


def legend_item(alg_name):
    color = ALG2COLOR.get(alg_name, "tab:blue")
    p1 = plt.plot(0, 0, color=color, linewidth=1)
    p2 = plt.fill(np.NaN, np.NaN, color, alpha=0.5)
    return (p2[0], p1[0])


def plot_returns_over_variable(
    episode_returns,
    x,
    xvar_name,
    alg_name,
    env_name,
    ax=None,
    color=None,
    show_ylabel=True,
):
    """Plot returns over some variable.

    Plot the episode returns over a variable (for instance, time in mu's, or
    n. of steps).

    params:
    - episode_returns: a np.ndarray containing returns, shape (n_episodes, n_runs).
    - x: a np.ndarray containing the x variable values, shape (n_episodes, n_runs).
    - xvar_name: the name of the x-variable.
    - alg_name: the name of the algorithm used.
    - env_name: the name of the environment used.
    - ax: the ax to plot on. default value is plt.gca().
    - color: the color to use. default is based on alg_name.
    - running_mean_n: how many samples should the running mean avg over.
    """
    # if xvar_name == "time":
    #     # convert x to us
    #     x = (x * 100000).round().astype(int)
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ALG2COLOR.get(alg_name, "tab:blue")
    # we want to compute mean and std over various runs
    nof_x = int(x.shape[0])
    mean_x = np.mean(x, axis=1)
    running_mean_n = int(nof_x / 30)
    episode_returns_running_mean = np.array(
        [running_mean(l, n=running_mean_n) for l in episode_returns.T]
    ).T

    mean_episode_returns = np.mean(episode_returns_running_mean, axis=1)
    std_episode_returns = np.std(episode_returns_running_mean, axis=1)

    # to smooth the plot, we use a running mean for the returns
    # running_mean_episode_returns = running_mean(
    #     mean_episode_returns, n=running_mean_n
    # )
    # running_std_episode_returns = running_mean(
    #     std_episode_returns, n=running_mean_n
    # )

    cumsum_x = np.cumsum(mean_x)[:-running_mean_n]

    # n_runs = x.shape[1]
    #
    # # we want to compute mean and std over various runs, divide by 0 so it's
    # # all nan
    # mean_x = np.mean(x, axis=1)
    # mean_episode_returns = np.mean(episode_returns, axis=1)
    # std_episode_returns = np.std(episode_returns, axis=1)
    # x2avg_episode_returns_running = np.array(
    #     [running_mean(l, n=running_mean_n) for l in x2avg_episode_returns.T]
    # ).T
    # mean_episode_returns = np.nanmean(episode_returns, axis=1)
    # std_episode_returns = np.nanstd(x2avg_episode_returns_running, axis=1)
    # # std_episode_returns_running = np.array(
    # #     [running_mean(l, n=running_mean_n) for l in x2avg_episode_returns.T]
    # # ).T

    # # to smooth the plot, we use a running mean for the returns

    # running_mean_episode_returns = running_mean(
    #     mean_episode_returns, n=running_mean_n
    # )
    # running_std_episode_returns = running_mean(
    #     std_episode_returns, n=running_mean_n
    # )

    # cumsum_x = np.arange(
    #     running_mean_episode_returns.size
    # )  # np.cumsum(mean_x)[:-running_mean_n]
    # if xvar_name == "time":
    #     # convert us to ms
    #     cumsum_x = cumsum_x.astype(int) / 100000
    ax.fill_between(
        cumsum_x,
        mean_episode_returns + std_episode_returns,
        mean_episode_returns - std_episode_returns,
        color=color,
        alpha=0.1,
        label=alg_name,
    )
    ax.plot(cumsum_x, mean_episode_returns, color=color, label=alg_name)
    if xvar_name == "time":
        ax.set_xlabel(f"{xvar_name} (ms)")
    else:
        ax.set_xlabel(xvar_name)
    if show_ylabel:
        ax.set_ylabel("Mean episodic return")


def plot_results_for_alg(
    episode_returns, xvar2results, alg_name, env_name, axs
):
    for i, (ax, (xvar_name, results)) in enumerate(
        zip(axs, xvar2results.items())
    ):
        plot_returns_over_variable(
            episode_returns,
            x=results,
            xvar_name=xvar_name,
            alg_name=alg_name,
            env_name=env_name,
            ax=ax,
            show_ylabel=(i == 0),
        )


def plot_results(
    alg2results, env_name, figsize=(12, 7), fname=None, running_mean_n=50
):
    """create grid of plots containing all results in alg2results.

    alg2results structure:
    ```
    alg2results = {
        "sarsa": {
            "episode_returns": returns_sarsa,
            "xvar2results": {
                "episode lengths": lengths_sarsa,
                "time": times_sarsa
            },
        },
        "expected sarsa": {
            "episode_returns": returns_esarsa,
            "xvar2results": {
                "episode lengths": lengths_esarsa,
                "time": times_esarsa
            },
        },
    }
    ```

    """
    xvars = list(list(alg2results.values())[0]["xvar2results"].keys())
    n_xvars = len(xvars)
    fig, axs = plt.subplots(1, n_xvars, figsize=figsize, sharey=True)

    for (alg_name, results) in alg2results.items():
        episode_returns = results["episode_returns"]
        plot_results_for_alg(
            episode_returns,
            results["xvar2results"],
            alg_name,
            env_name,
            axs,
        )

    legend_ary = [
        list(elem)
        for elem in (
            zip(
                *[
                    (legend_item(alg_name), alg_name)
                    for alg_name in alg2results.keys()
                ]
            )
        )
    ]
    axs[-1].legend(*legend_ary)

    plt.subplots_adjust(
        left=0.08, bottom=0.09, right=0.92, top=0.93, wspace=0.1, hspace=0.4
    )
    plt.suptitle(
        f"Mean episode return over {join_with_final_and(xvars)}, in the {env_name} env."
    )

    if fname is None:
        plt.show()
    else:
        # plt.savefig(fname, bbox_inches="tight_layout")
        plt.savefig(fname)


# from src.envs.windy_gridworld import WindyGridworldEnv
# from src.sarsa import sarsa, expected_sarsa

# env = WindyGridworldEnv()


# def get_results(fn, n_runs=3, n_episodes=1000):
#     returns = np.zeros((n_episodes, n_runs))
#     lengths = np.zeros((n_episodes, n_runs))
#     times = np.zeros((n_episodes, n_runs))
#     for r in range(n_runs):
#         (
#             _,
#             (episode_lengths, episode_returns, episode_times),
#             _,
#         ) = fn(env, n_episodes)
#         returns[:, r] = episode_returns
#         lengths[:, r] = episode_lengths
#         times[:, r] = episode_times
#     return {
#         "episode_returns": returns,
#         "xvar2results": {"episode lengths": lengths, "time": times},
#     }


# alg2results = {
#     "sarsa": get_results(sarsa),
#     "expected sarsa": get_results(expected_sarsa),
# }

# plt.close("all")
# plot_results(
#     alg2results,
#     "windygridworld",
# )
