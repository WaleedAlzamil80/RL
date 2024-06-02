import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import numba as nb
import os
import cv2
import gymnasium as gym

# ls videos/*.mp4 | sort -V | sed "s/^/file '/;s/$/'/" > videos.txt
# ffmpeg -f concat -safe 0 -i videos.txt -c copy output.mp4

def query_env(name, con = False):
    env = gym.make(name)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")
    if con:
        return env.observation_space.shape[0], env.action_space.shape[0]
    return env.observation_space.shape[0], env.action_space.n

class policy_network(nn.Module):
    def __init__(self, obs_space_dim: int, action_space_dim: int, continous = False):
        super().__init__()

        self.fc1 = nn.Linear(obs_space_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if continous:
            self.mu = nn.Linear(128, action_space_dim)
            self.log_std = nn.Linear(128, action_space_dim)
        else:
            self.fc4 = nn.Linear(128, action_space_dim)

        self.softmax = nn.Softmax(dim = -1)
        self.continous = continous

    def forward(self, X):
        X = self.tanh(self.fc1(X))
        X = self.tanh(self.fc2(X))
        X = self.tanh(self.fc3(X))
        if self.continous:
            mu = self.mu(X)
            log_std = self.log_std(X)
            return mu, log_std

        X = self.softmax(self.fc4(X))

        return X

class REINFORCE():
    def __init__(self, obs_space_dim, action_space_dim, reward_norm = False, continous = False):
        self.logprobs = []
        self.rewards = []
        self.loss = []
        self.gamma = 0.99
        self.lr = 1e-4
        self.eps = 1e-6
        self.continous = continous
        self.reward_norm = reward_norm

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.policy = policy_network(obs_space_dim, action_space_dim, continous = continous).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.lr)


    def sample_action(self, state):
        state = torch.tensor(state).to(self.device)
        if self.continous:
            mu, logstd = self.policy(state)
            action_dist = torch.distributions.Normal(mu, logstd.exp())
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()
            self.logprobs.append(log_prob)
            return self.policy.tanh(action).cpu().detach().numpy()

        action_probs = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.logprobs.append(action_dist.log_prob(action))
        return action.item()

    def update(self):
        running_gs = 0
        Gs = []
        for R in self.rewards[::-1]:
            running_gs = running_gs + R * self.gamma
            Gs.insert(0, running_gs)

        Gs = torch.tensor(Gs).to(self.device)
        if self.reward_norm:
            Gs = (Gs - Gs.mean()) / (Gs.std() + self.eps)

        loss = 0
        for log_prob, reward in zip(self.logprobs, Gs):
            loss += -log_prob * reward
        self.loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logprobs = []
        self.rewards = []

    def train(self, num_episoeds, save_every = 100):
        pass