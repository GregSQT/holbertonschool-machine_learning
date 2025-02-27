#!/usr/bin/env python3
"""Let a pre-trained agent play a full game of Atari's Breakout (using BreakoutNoFrameskip-v4)
with suppressed warnings."""
import os
import warnings

# Suppress TensorFlow logging (set to '2' to hide info and warning messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Suppress Python warnings (you can adjust the filter as needed)
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

# Compatibility wrapper for Gymnasium
class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

    def render(self, **kwargs):
        return self.env.render()

# Atari preprocessor (same as before)
class AtariProcessor(Processor):
    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        # Ensure observation is 3D (height, width, channels)
        assert observation.ndim == 3, f"Expected observation.ndim == 3, got {observation.ndim}"
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        processed = np.array(img)
        assert processed.shape == (84, 84), f"Processed shape mismatch: {processed.shape}"
        return processed.astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def build_model(num_actions):
    input_shape = (4, 84, 84)
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model

if __name__ == '__main__':
    # Use BreakoutNoFrameskip-v4 for full game episodes (loss of one life does not end the episode)
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    # Set a default render fps to avoid inconsistent fps warnings
    if env.metadata.get('render_fps') is None:
        env.metadata['render_fps'] = 30
    env = CompatibilityWrapper(env)
    env.reset()

    num_actions = env.action_space.n
    window = 4

    model = build_model(num_actions)
    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps',
        value_max=1.0, value_min=0.1, value_test=0.05,
        nb_steps=1000000
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.
    )
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    
    # Load pre-trained weights if available.
    try:
        dqn.load_weights('policy.h5')
        print("Loaded pre-trained weights.")
    except Exception as e:
        print("Failed to load weights. Exception:", e)
    
    try:
        print("Testing for 10 episodes ...")
        dqn.test(env, nb_episodes=10, visualize=True)
    except KeyboardInterrupt:
        print("Testing interrupted by user.")
