#!/usr/bin/env python3
"""
a function that load environment of frozen lake that is already
pre made env from gymnasium
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False, render_mode=None):
    """
    desc: list of lists containing a custom description of the map to
    load for the environment
    map_name: string containing the pre-made map to load
    is_slippery: boolean to determine if the ice is slippery
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi")
    return env
