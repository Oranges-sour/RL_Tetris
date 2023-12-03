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
        self.fc1 = nn.Linear(
            7 * 3 + 10 + 10 + 1,
            128,
            device=device,
        )
        self.batchnorm4 = nn.BatchNorm1d(128, device=device)

        self.fc2 = nn.Linear(128, 128, device=device)
        self.batchnorm5 = nn.BatchNorm1d(128, device=device)

        self.fc2_v = nn.Linear(128, 1, device=device)

        self.relu = nn.ReLU()

    def forward(self, x, x2):
        x2 = x2.reshape(-1, 7 * 3 + 10 + 10 + 1)

        x = x2

        x = self.relu(self.batchnorm4(self.fc1(x)))

        x = self.relu(self.batchnorm5(self.fc2(x)))

        V = self.fc2_v(x)

        return V


# Dueling Q Network
class Network1(nn.Module):
    def __init__(self) -> None:
        super(Network1, self).__init__()

        self.out_channel1 = 16
        self.out_channel2 = 32
        self.out_channel3 = 48

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channel1,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.batchnorm1 = nn.BatchNorm2d(self.out_channel1, device=device)
        # self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channel1,
            out_channels=self.out_channel2,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.batchnorm2 = nn.BatchNorm2d(self.out_channel2, device=device)

        self.conv3 = nn.Conv2d(
            in_channels=self.out_channel2,
            out_channels=self.out_channel3,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.batchnorm3 = nn.BatchNorm2d(self.out_channel3, device=device)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            7 * 3 + 10 + 10 + self.out_channel3 * int(WW / 2) * int(HH / 2),
            1024,
            device=device,
        )
        self.batchnorm4 = nn.BatchNorm1d(1024, device=device)

        self.fc2 = nn.Linear(1024, 1024, device=device)
        self.batchnorm5 = nn.BatchNorm1d(1024, device=device)

        self.fc2_v = nn.Linear(1024, 1, device=device)

        self.relu = nn.ReLU()

    def forward(self, x, x2):
        x = x.reshape(-1, 1, HH, WW)

        x = self.relu(self.batchnorm1(self.conv1(x)))

        x = self.relu(self.batchnorm2(self.conv2(x)))

        x = self.relu(self.batchnorm3(self.conv3(x)))

        x = self.pool3(x)

        x = x.reshape(-1, self.out_channel3 * int(WW / 2) * int(HH / 2))

        x2 = x2.reshape(-1, 7 * 3 + 10 + 10)

        x = torch.cat((x, x2), dim=1)

        x = self.relu(self.batchnorm4(self.fc1(x)))

        x = self.relu(self.batchnorm5(self.fc2(x)))

        V = self.fc2_v(x)

        return V
