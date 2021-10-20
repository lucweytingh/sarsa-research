import time
from tqdm import tqdm as _tqdm

from src.utils import init_Q, EpsilonGreedyPolicy, get_actions


def train_sarsa(
    env,
    num_episodes,
    Q,
    policy,
    get_Q_for_next_sa,
    discount_factor=1.0,
    alpha=0.5,
):
    """Shared training logic for SARSA or Expected SARSA.

    SARSA and Expected SARSA only differ in the way that the Q-value based on
    the next state is computed. Hence, this logic is represented by the
    get_Q_for_next_sa function, which is passed by SARSA or Expected SARSA.
    """

    # keep track of useful statistics over training
    stats = []
    diffs = []

    for i_episode in range(num_episodes):
        state = env.reset()
        start_time = time.time()
        # the current timestep
        t = 0
        # keep track of the discounted sum of rewards
        R = 0
        # we need an initial action
        action = policy.sample_action(state)
        while True:
            # perform step in the environment
            (new_state, reward, done, _) = env.step(action)
            # compute a new action
            new_action = policy.sample_action(new_state)
            R += reward * (discount_factor ** t)
            Q_sa = Q.get(state, action)

            # compute new Q[s, a] value
            updated_Q_sa = Q_sa + alpha * (
                reward
                + discount_factor * get_Q_for_next_sa(Q, new_state, new_action)
                - Q_sa
            )
            # store that value in Q-function
            Q.set(state, action, updated_Q_sa)

            # move on to next timestep
            state = new_state
            action = new_action
            t += 1

            if done:
                break
        duration = time.time() - start_time
        stats.append((t, R, duration))

    episode_lengths, episode_returns, episode_times = zip(*stats)
    return Q, (episode_lengths, episode_returns, episode_times), diffs


def sarsa(
    env,
    num_episodes,
    discount_factor=1.0,
    alpha=0.5,
):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy
    policy.

    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its
        sample_action method.
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

    def get_Q_for_next_sa(Q, new_state, new_action):
        return Q.get(new_state, new_action)

    return train_sarsa(
        env=env,
        num_episodes=num_episodes,
        Q=Q,
        policy=policy,
        get_Q_for_next_sa=get_Q_for_next_sa,
        discount_factor=discount_factor,
        alpha=alpha,
    )


def expected_sarsa(
    env,
    num_episodes,
    discount_factor=1.0,
    alpha=0.5,
):
    """
    expected SARSA algorithm: On-policy TD control. Finds the optimal
    epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its
        sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
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

    actions = get_actions(env)
    non_greedy_action_prob = policy.epsilon / len(list(actions))
    greedy_action_prob = ((1 - policy.epsilon)) + non_greedy_action_prob

    def policy_prob(s, a, best_a):
        "return probability of a being chosen from s under policy."
        return greedy_action_prob if a == best_a else non_greedy_action_prob

    def get_Q_for_next_sa(Q, new_state, new_action):
        best_a = Q.get_best_action(new_state)
        expected_q = sum(
            policy_prob(new_state, a, best_a) * Q.get(new_state, a)
            for a in actions
        )
        return expected_q

    return train_sarsa(
        env=env,
        num_episodes=num_episodes,
        Q=Q,
        policy=policy,
        get_Q_for_next_sa=get_Q_for_next_sa,
        discount_factor=discount_factor,
        alpha=alpha,
    )


NAME2ALG = {"expected_sarsa": expected_sarsa, "sarsa": sarsa}


# from windy_gridworld import WindyGridworldEnv
# import matplotlib.pyplot as plt
# import numpy as np

# env = WindyGridworldEnv()


# def running_mean(vals, n=1):
#     cumvals = np.array(vals).cumsum()
#     return (cumvals[n:] - cumvals[:-n]) / n


# (
#     Q_sarsa,
#     (episode_lengths_sarsa, episode_returns_sarsa, episode_times_sarsa),
#     diffs,
# ) = sarsa(env, 1000)

# (
#     Q_sarsa,
#     (episode_lengths_e_sarsa, episode_returns_e_sarsa, episode_times_e_sarsa),
#     diffs,
# ) = expected_sarsa(env, 1000)

# print(len(episode_lengths_sarsa))
# n = 50
# # We will help you with plotting this time
# plt.clf()
# plt.plot(
#     np.cumsum(episode_times_sarsa[:-n]),
#     running_mean(episode_returns_sarsa, n),
#     label="sarsa",
# )
# plt.plot(
#     np.cumsum(episode_times_e_sarsa[:-n]),
#     running_mean(episode_returns_e_sarsa, n),
#     label="expected_sarsa",
# )
# plt.title("Return attained during training ")
# plt.xlabel("Time")
# plt.ylabel("Return")
# plt.legend()
# plt.show()
