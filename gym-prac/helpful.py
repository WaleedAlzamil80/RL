import gymnasium as gym
from gymnasium.spaces import Box

import torch
import torch.nn as nn
import numpy as np
import numba as nb

import cv2
import matplotlib.pyplot as plt

import os
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
        # self.fc5 = nn.Linear(512, 512)

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
        # x = self.relu(self.fc5(x))

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
    def __init__(self, obs_space_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_space_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc = nn.Linear(512, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc(x)

        return x

class REINFORCE:
    """
    REINFORCE algorithm implementation.
    """
    def __init__(self, env_info, reward_norm = False):
        self.logprobs = []
        self.rewards = []
        self.losses = []
        self.gamma = 0.99
        self.lr = 1e-4
        self.eps = 1e-6
        self.continous = env_info["action_space_continuous"]
        self.reward_norm = reward_norm

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.policy = PolicyNetwork(env_info).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        if self.continous:
            mu, log_std = self.policy(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            self.logprobs.append(log_prob)
            return torch.tanh(action).cpu().detach().numpy()

        action_probs = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.logprobs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        running_gs = 0
        Gs = []
        for R in self.rewards[::-1]:
            running_gs = running_gs * self.gamma + R
            Gs.insert(0, running_gs)

        Gs = torch.tensor(Gs, dtype=torch.float32).to(self.device)
        if self.reward_norm:
            Gs = (Gs - Gs.mean()) / (Gs.std() + self.eps)

        loss = 0
        for log_prob, G in zip(self.logprobs, Gs):
            loss += -log_prob * G

        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logprobs = []
        self.rewards = []

    def combined_episode_videos(self,to_save_at):
        video_names = np.asarray(glob(self.params.export_video+'/*.mp4'))
        if video_names.shape[0]==0: return
        cap = cv2.VideoCapture(video_names[0])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(to_save_at+'/combined_{}.mp4'.format(self.params.env_name),cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
        for idx,video in enumerate(video_names):
            episode_idx = idx*self.params.log_episode_interval
            cap = cv2.VideoCapture(video)
            counter = 0
            while(True):
                font_size = self.params.font_size*(frame_width/600)
                if font_size<0.5:
                    font_size = 0.5
                margin = int(self.params.font_margin/600*frame_width)
                # Capture frames in the video
                ret, frame = cap.read()

                if not ret:
                    break
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame,'Episode: {}'.format(episode_idx+1),(margin, margin),font, font_size,self.params.font_color,2, cv2.LINE_4)
                cv2.putText(frame,'Reward: {:.2f}'.format(self.episode_rewards[episode_idx]), (margin, frame_height-margin), font, font_size,self.params.font_color, 2,cv2.LINE_4)
                out.write(frame)
                counter += 1

            cap.release()
        out.release()

    def save_nets(self,pth_name):
        torch.save(self.policy.state_dict(), f"{pth_name}_policy_net.pth")

class ActorCritic:
    """
    Actor-Critic algorithm implementation.
    """
    def __init__(self, obs_space_dim, action_space_dim, reward_norm=False, continous=False):
        self.logprobs = []
        self.rewards = []
        self.losses = []
        self.gamma = 0.99
        self.lr = 1e-4
        self.eps = 1e-6
        self.continous = continous
        self.reward_norm = reward_norm

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.policy = PolicyNetwork(obs_space_dim, action_space_dim, continous=continous).to(self.device)
        self.value = ValueNetwork(obs_space_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        R = self.value(state).detach().cpu().item()
        self.rewards.append(R)

        if self.continous:
            mu, log_std = self.policy(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            self.logprobs.append(log_prob)
            return torch.tanh(action).cpu().detach().numpy()

        action_probs = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.logprobs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        running_gs = 0
        Gs = []
        for R in self.rewards[::-1]:
            running_gs = running_gs * self.gamma + R
            Gs.insert(0, running_gs)

        Gs = torch.tensor(Gs, dtype=torch.float32).to(self.device)
        if self.reward_norm:
            Gs = (Gs - Gs.mean()) / (Gs.std() + self.eps)

        loss = 0
        for log_prob, G in zip(self.logprobs, Gs):
            loss += -log_prob * G

        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logprobs = []
        self.rewards = []

    def combined_episode_videos(self,to_save_at):
        video_names = np.asarray(glob(self.params.export_video+'/*.mp4'))
        if video_names.shape[0]==0: return
        cap = cv2.VideoCapture(video_names[0])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(to_save_at+'/combined_{}.mp4'.format(self.params.env_name),cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
        for idx,video in enumerate(video_names):
            episode_idx = idx*self.params.log_episode_interval
            cap = cv2.VideoCapture(video)
            counter = 0
            while(True):
                font_size = self.params.font_size*(frame_width/600)
                if font_size<0.5:
                    font_size = 0.5
                margin = int(self.params.font_margin/600*frame_width)
                # Capture frames in the video
                ret, frame = cap.read()

                if not ret:
                    break
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame,'Episode: {}'.format(episode_idx+1),(margin, margin),font, font_size,self.params.font_color,2, cv2.LINE_4)
                cv2.putText(frame,'Reward: {:.2f}'.format(self.episode_rewards[episode_idx]), (margin, frame_height-margin), font, font_size,self.params.font_color, 2,cv2.LINE_4)
                out.write(frame)
                counter += 1

            cap.release()
        out.release()

    def save_nets(self,pth_name):
        torch.save(self.policy.state_dict(), f"{pth_name}_policy_net.pth")
        torch.save(self.value.state_dict(), f"{pth_name}_value_net.pth")