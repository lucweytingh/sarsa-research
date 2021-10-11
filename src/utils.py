import numpy as np


def init_Q(env):
    "initialize empty Q table given the environment"
    return np.zeros((env.nS, env.nA))
