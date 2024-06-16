import gymnasium as gym
import numpy as np
import cv2
import os
import time
from helpful import *

from pyvirtualdisplay import Display
from gymnasium.wrappers import RecordVideo

# https://github.com/openai/gym/wiki/Leaderboard
# https://gymnasium.farama.org/
env_name = "BipedalWalker-v3"
continous = True
obs_dim, act_dim = query_env(env_name, continous)

env = gym.make(env_name, render_mode = 'rgb_array') #  LunarLanderContinuous-v2  "BipedalWalker-v3" Hardcore continous
display = Display(visible=0, size=(1400, 900))
display.start()

env.metadata['render_fps'] = 30

# Setup the wrapper to record the video
env = RecordVideo(env, video_folder='./videos', episode_trigger=lambda episode_id: (episode_id % 30) == 0)

agent = REINFORCE(obs_dim, act_dim, continous = continous)

episodeNumper = 3000
for episode_index in range(episodeNumper):
    observation, _ = env.reset()
    done = False
    while not done:
        action = agent.sample_action(observation)
        if continous:
            observation, reward, done, truncated, info = env.step(action)
        else:
            observation, reward, done, truncated, info = env.step(int(action))
        agent.rewards.append(reward)
        done = done or truncated

    if (episode_index) % 100 == 0:
        print(f"Episode {episode_index + 1} : " , "Reward : ", int(np.sum(np.array(agent.rewards))))
    agent.update()

env.close
agent.save_nets(env_name)
agent.combined_episode_videos(env_name)