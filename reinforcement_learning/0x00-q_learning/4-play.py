#!/usr/bin/env python3
"""
play with the model trained with Q-learning
based on:
https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    has the trained agent play an episode
    Args:
        env: is the FrozenLakeEnv instance
        Q: is a numpy.ndarray containing the Q-table
        max_steps: is the maximum number of steps in the episode

    Returns: the total rewards for the episode

    """
    state = env.reset()
    env.render()
    done = False
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        env.render()

        if done:
            # check if the agent reached the goal(G) or fell into a hole(H)
            break

        state = new_state
    # close the connection to the environment
    env.close()

    return reward
