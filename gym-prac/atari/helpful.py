import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from gymnasium.envs import mujoco

import torch
import torch.nn as nn
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import cv2
import os
import time
import glob

# ls videos/*.mp4 | sort -V | sed "s/^/file '/;s/$/'/" > videos.txt
# ffmpeg -f concat -safe 0 -i videos.txt -c copy output.mp4


def query_env(name):
    """
    Query environment details and return the dimensions of the observation and action spaces,
    including state space, action space, input shape, output shape, and the range of control actions.
    """
    env = gym.make(name)
    spec = gym.spec(name)
    
    # Extracting details
    action_space = env.action_space
    observation_space = env.observation_space
    action_space_details = {
        "shape": action_space.shape,
        "dtype": action_space.dtype,
        "high": action_space.high if hasattr(action_space, 'high') else None,
        "low": action_space.low if hasattr(action_space, 'low') else None
    }
    observation_space_details = {
        "shape": observation_space.shape,
        "dtype": observation_space.dtype,
        "high": observation_space.high if hasattr(observation_space, 'high') else None,
        "low": observation_space.low if hasattr(observation_space, 'low') else None
    }
    
    # Determine if spaces are continuous or discrete
    action_space_continuous = isinstance(action_space, Box)
    observation_space_continuous = isinstance(observation_space, Box)
    
    # Displaying details
    print(f"Action Space: {action_space}")
    print(f"Observation Space: {observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")
    print(f"Action Space Details: {action_space_details}")
    print(f"Observation Space Details: {observation_space_details}")
    print(f"Action Space Continuous: {action_space_continuous}")
    print(f"Observation Space Continuous: {observation_space_continuous}")
    
    if action_space_continuous:
        output_shape = action_space.shape[0]
    else:
        output_shape = action_space.n

    if observation_space_continuous:
        input_shape = observation_space.shape[0]
    else:
        input_shape = observation_space.n
    
    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "action_space_details": action_space_details,
        "observation_space_details": observation_space_details,
        "action_space_continuous": action_space_continuous,
        "observation_space_continuous": observation_space_continuous
    }