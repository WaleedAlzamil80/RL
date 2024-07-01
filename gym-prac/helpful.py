import gymnasium as gym
from gymnasium.spaces import Box
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

class PolicyNetwork(nn.Module):
    """
    Neural network for policy approximation in REINFORCE algorithm.
    """
    def __init__(self, env_info):
        super(PolicyNetwork, self).__init__()
        obs_dim, act_dim, continous_action, continous_state = env_info["input_shape"], env_info["output_shape"], env_info["action_space_continuous"], env_info["observation_space_continuous"]

        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 512)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.continous_action = continous_action

        if continous_action:
            self.mu = nn.Linear(512, act_dim)
            self.log_std = nn.Linear(512, act_dim)
        else:
            self.fc = nn.Linear(512, act_dim)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        if self.continous_action:
            mu = self.mu(x)
            log_std = self.log_std(x)
            return mu, log_std

        x = self.softmax(self.fc(x))
        return x


class ValueNetwork(nn.Module):
    """
    Neural network for value approximation in Actor-Critic algorithm.
    """
    def __init__(self, env_info):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(env_info["input_shape"], 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc(x)

        return x