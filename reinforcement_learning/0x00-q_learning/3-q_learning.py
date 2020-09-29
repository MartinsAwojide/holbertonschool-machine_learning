#!/usr/bin/env python3
"""
performs Q-learning
based on:
https://www.youtube.com/watch?v=HGeI30uATws&feature=youtu.
be&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&t=24
and
https://www.kaggle.com/rajmiglani/rl-frozen-lake
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs Q-learning
    Args:
        env: is the FrozenLakeEnv instance
        Q: is a numpy.ndarray containing the Q-table
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
        epsilon: is the initial threshold for epsilon greedy
        min_epsilon: is the minimum value that epsilon should decay to
        epsilon_decay: is the decay rate for updating epsilon between episodes

    When the agent falls in a hole, the reward would be updated to be -1
    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode

    """
    rewards = []

    # let the agent play for defined number of episodes
    for episode in range(episodes):
        # reset the environment for each episode
        state = env.reset()
        # define initial parameters
        step = 0
        # to keep track whether the agent dies
        done = False
        # keep track of rewards at each episode
        total_rewards = 0

        # run for each episode
        for step in range(max_steps):
            # Exploration-explotation trade-off

            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            # Update Q-table for Q(s, a)
            part = (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            Q[state, action] += alpha * part
            state = new_state

            if done is True:
                if reward == 0.0:
                    # if falls in a hole or lost last live
                    total_rewards = -1
                total_rewards += reward
                break

            total_rewards += reward
        rewards.append(total_rewards)
        part = (1 - min_epsilon) * np.exp(-epsilon_decay * episode)
        epsilon = min_epsilon + part
    return Q, rewards
