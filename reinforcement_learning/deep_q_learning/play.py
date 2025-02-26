#!/usr/bin/env python3
"""play an agent that can play Atari's Breakout"""

import os
import logging
import time
import cv2

# Suppress TensorFlow messages (only errors will be shown)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
# Disable eager execution (using TF1 compatibility for keras-rl)
tf.compat.v1.disable_eager_execution()

import gymnasium as gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import tensorflow.keras as K

# Import build_model and AtariProcessor from your training script.
build_model = __import__('train').build_model
AtariProcessor = __import__('train').AtariProcessor

# Compatibility wrapper that converts Gymnasium's API (5-tuple) to the 4-tuple expected by keras-rl.
class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

    def render(self, **kwargs):
        # Remove the 'mode' argument if present to avoid errors.
        kwargs.pop('mode', None)
        frame = self.env.render(**kwargs)
        if frame is not None:
            cv2.imshow("Breakout", frame)
            cv2.waitKey(1)  # Update the window
        return frame

if __name__ == '__main__':
    # Create the environment with render_mode 'rgb_array' so that render() returns a frame.
    env = gym.make("Breakout-v4", render_mode='rgb_array')
    env.metadata['render_fps'] = 30  # Set fps metadata to avoid warnings.
    env = CompatibilityWrapper(env)
    env.reset()

    num_actions = env.action_space.n
    model = build_model(num_actions)
    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = AtariProcessor()

    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   processor=processor, memory=memory)
    dqn.compile(K.optimizers.legacy.Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    print("Testing for 10 episodes ...")
    dqn.test(env, nb_episodes=10, visualize=True)
    
    cv2.destroyAllWindows()
