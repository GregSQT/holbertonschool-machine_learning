#!/usr/bin/env python3
"""Entraîne un agent pour jouer à Breakout (Atari)"""

from PIL import Image
import numpy as np
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
# Utilisation de l'optimiseur legacy pour éviter l'erreur get_updates
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


# Wrapper de compatibilité pour Gymnasium
class CompatibilityWrapper(gym.Wrapper):
    def step(self, action):
        # Gymnasium renvoie : observation, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

# Préprocesseur pour Atari
class AtariProcessor(Processor):
    def process_observation(self, observation):
        # Si observation est un tuple (observation, info), on extrait l'observation
        if isinstance(observation, tuple):
            observation = observation[0]
        # On s'assure que l'observation est en 3 dimensions (hauteur, largeur, canaux)
        assert observation.ndim == 3, f"Expected observation.ndim == 3, got {observation.ndim}"
        img = Image.fromarray(observation)
        # Redimensionnement à 84x84 et conversion en niveaux de gris
        img = img.resize((84, 84)).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84), f"Processed shape mismatch: {processed_observation.shape}"
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def build_model(num_actions):
    """Construit le modèle de réseau de neurones."""
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
    # Création de l'environnement avec render_mode="human" pour l'affichage
    env = gym.make("Breakout-v4", render_mode="human")
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

    # IMPORTANT : On réduit le nb_steps_warmup à 1000 pour que l'agent commence à apprendre
    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=1000,
        gamma=0.99,
        target_model_update=1000,
        train_interval=4,
        delta_clip=1.
    )
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    
    # Entraînement sur 17 500 étapes (attention : pour Breakout, il faut généralement des millions d'étapes)
    dqn.fit(env,
            nb_steps=1000000,
            log_interval=1000,
            visualize=False,
            verbose=2)

    dqn.save_weights('policy.h5', overwrite=True)
