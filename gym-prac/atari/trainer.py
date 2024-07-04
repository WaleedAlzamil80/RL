import argparse

import gymnasium as gym
import torch
import torch.nn as nn
from Deep_Q_CNN import DQN
from train_cnn import train
from helpful import *

import numpy as np

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="specify the environment to train on.")
env_name = "SpaceInvaders-v4"
env_info = query_env(env_name)