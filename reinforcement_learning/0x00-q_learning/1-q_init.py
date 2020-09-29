#!/usr/bin/env python3
"""
initializes the Q-table
based on:
https://www.learndatasci.com/tutorials/
reinforcement-q-learning-scratch-python-openai-gym/
"""
import numpy as np


def q_init(env):
    """
    initializes the Q-table
    Args:
        env: is the FrozenLakeEnv instance

    Returns: the Q-table as a numpy.ndarray of zeros

    """

    # create the Q-table with 0-load_env dimensions and filled with zeros
    # intially as the agent is unaware of the environment
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    return Q_table
