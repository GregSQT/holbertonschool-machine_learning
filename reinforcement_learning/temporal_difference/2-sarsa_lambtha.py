import gym
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) algorithm to update Q table.

    Args:
        env: The OpenAI Gym environment instance.
        Q: A numpy.ndarray of shape (s,a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: Total number of episodes to train over (default is 5000).
        max_steps: Maximum number of steps per episode (default is 100).
        alpha: The learning rate (default is 0.1).
        gamma: The discount rate (default is 0.99).
        epsilon: The initial threshold for epsilon-greedy strategy (default is 1).
        min_epsilon: The minimum value that epsilon should decay to (default is 0.1).
        epsilon_decay: The decay rate for updating epsilon between episodes (default is 0.05).

    Returns:
        Q: The updated Q table.
    """
    # Initialize eligibility traces
    z = np.zeros_like(Q)
    
    for iteration in range(episodes):
        state = env.reset()
        # Select an action using epsilon-greedy strategy
        action = np.argmax(np.random.uniform(size=Q.shape[1]) < epsilon)
        
        for t in range(max_steps):
            next_state, reward, done, iteration = env.step(action)
            
            # Choose the next action using epsilon-greedy strategy
            next_action = np.argmax(np.random.uniform(size=Q.shape[1]) < epsilon)
            
            # Calculate the target Q-value
            target = reward + gamma * Q[next_state, next_action]
            
            # Update the Q-value and eligibility traces
            delta = target - Q[state, action]
            Q[state, action] += alpha * delta
            z *= lambtha * gamma
            z[state, action] += 1
            
            # Decay eligibility traces
            z *= lambtha
            
            # Move to the next state and action
            state = next_state
            action = next_action
            
            # Check if the episode has ended
            if done:
                break
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    
    return Q