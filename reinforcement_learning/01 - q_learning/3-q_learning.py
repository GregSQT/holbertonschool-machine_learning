#!/usr/bin/env python3
"""
q learning function that works with the FrozenLakeEnv environment
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
        env,
        Q,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """
    env: FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: epsilon is a strategy from epsilon greedy to decide it's either 
        to explore or exploit
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: decay rate for updating epsilon between episodes
    Returns: Q, total_rewards
    """
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        rewards_current_episode = 0

        for step in range(max_steps):
            """get the action to take"""
            action = epsilon_greedy(Q, state, epsilon)
            """extract the new state, reward, done, and truncated"""
            new_state, reward, done, truncated, _ = env.step(action)
            if done and reward == 0:
                reward = -1
            """this here is the formula for updating the Q-table"""
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state] - Q[state, action]))
            state = new_state
            rewards_current_episode += reward
            """if the episode is done or truncated, break"""
            if done or truncated:
                break
        """epsilon decay for the next episode"""
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards
