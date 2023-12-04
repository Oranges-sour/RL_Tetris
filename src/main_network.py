import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np

# import gym
import time
import os

import pygame

import math

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

from tetris import Tetris

WW = Tetris.W - 4
HH = Tetris.H - 3



# Dueling Q Network
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()

        self.out_channel1 = 16
        self.out_channel2 = 32

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channel1,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.batchnorm1 = nn.BatchNorm2d(self.out_channel1, device=device)


        self.conv2 = nn.Conv2d(
            in_channels=self.out_channel1,
            out_channels=self.out_channel2,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.batchnorm2 = nn.BatchNorm2d(self.out_channel2, device=device)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            7 * 3 + 10 + 10 + self.out_channel2 * int(WW / 2) * int(HH / 2),
            512,
            device=device,
        )
        self.batchnorm4 = nn.BatchNorm1d(512, device=device)

        self.fc2 = nn.Linear(512, 256, device=device)
        self.batchnorm5 = nn.BatchNorm1d(256, device=device)

        self.fc2_v = nn.Linear(256, 1, device=device)

        self.relu = nn.ReLU()

    def forward(self, x, x2):
        x = x.reshape(-1, 1, HH, WW)

        x = self.relu(self.batchnorm1(self.conv1(x)))

        x = self.relu(self.batchnorm2(self.conv2(x)))

        x = self.pool1(x)

        x = x.reshape(-1, self.out_channel2 * int(WW / 2) * int(HH / 2))

        x2 = x2.reshape(-1, 7 * 3 + 10 + 10)

        x = torch.cat((x, x2), dim=1)

        x = self.relu(self.batchnorm4(self.fc1(x)))

        x = self.relu(self.batchnorm5(self.fc2(x)))

        V = self.fc2_v(x)

        return V
