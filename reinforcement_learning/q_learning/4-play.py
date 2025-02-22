#!/usr/bin/env python3
"""
function having the trained agent to play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    * env is the FrozenLakeEnv instance
    * Q is a numpy.ndarray containing the Q-table
    * max_steps is the maximum number of steps in the episode
    * Each state of the board should be displayed via the console
    * You should always exploit the Q-table
    * Returns: the total rewards for the episode
    """
    state, _ = env.reset()
    rendered_outputs = []
    total_reward = 0
    rendered_outputs.append(env.render())
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _, _ = env.step(action)
        rendered_outputs.append(env.render())
        total_reward += reward
        if done:
            break
    return total_reward, rendered_outputs
