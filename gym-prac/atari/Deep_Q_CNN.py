import torch
import torch.nn as nn
import numpy as np 

class DQN(nn.Module):
    def __init__(self, env_info):
        super(DQN, self).__init__()
        self.input_shape = env_info["input_shape"]
        self.output_shape = env_info["output_shape"]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(10, 6), stride=(2, 2), padding=(0, 0)) # (210, 160) ---> ((Win - k + 2p) / s) + 1   ------> (101, 78)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(7, 4), stride=(2, 2), padding=(0, 0)) # (101, 78) ---> (48, 38)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(6, 2), stride=(2, 2), padding=(0, 0)) # (48, 38) ---> (22, 19)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(6, 3), stride=(2, 2), padding=(0, 0)) # (22, 19) ---> (9, 9)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)) # (9, 9) ---> (7, 7)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)) # (7, 7) ---> (5, 5)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)) # (5, 5) ---> (3, 3)
        self.conv8 = nn.Conv2d(2048, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)) # (3, 3) ---> (1, 1)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, self.output_shape)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
        self.net1 = nn.Sequential(
            self.conv1, self.relu, self.conv2, self.relu,
            self.conv3, self.relu, self.conv4, self.relu,
            self.conv5, self.relu, self.conv6, self.relu,
            self.conv7, self.relu, self.conv8, 
        )
        self.net2 = nn.Sequential(
            self.fc1, self.fc2,
            self.fc3, self.fc4
        )
    def forward(self, x):
        pass