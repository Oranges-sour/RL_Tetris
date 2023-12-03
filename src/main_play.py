import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy


# import gym
import time
import os

import pygame

import math


from tetris import Tetris
from tetris import get_all_possible_state

from main_network import Network


WW = Tetris.W - 4
HH = Tetris.H - 3

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((WW * 20, HH * 20))
screen.fill((156, 156, 156))
pygame.display.set_caption("main")

img = []
for i in range(1, 6):
    img.append(
        pygame.transform.scale(pygame.image.load(f"blocks/{i}.jpg"), (20, 20)).convert()
    )


# cpu训练
device = "cpu"

# 游戏环境
env = Tetris(1)


# 观察空间
obser_n = WW * HH


render_game = True


# 双网络交替
network = Network()


network = torch.load("model/1699627378_500.pth")
network.eval()


# 游戏次数
episode = 12

# 一个episode的游戏最大步数
game_step = 3000
###########################


def state_tensor(s):
    x = torch.from_numpy(s).to(device=device, dtype=torch.float32)
    return x


def func_render_game(sss):
    if render_game == True:
        for event in pygame.event.get():
            # 判断用户是否点了关闭按钮
            if event.type == pygame.QUIT:
                # 卸载所有模块
                pygame.quit()

        screen.fill((0, 0, 0))
        ss = sss.get_colored_map()

        for x in range(0, HH):
            for y in range(0, WW):
                if ss[x][y] == 0:
                    continue
                screen.blit(img[int(ss[x][y]) - 1], (y * 20, x * 20))

        pygame.display.flip()


def play():
    global env
    global e
    global running_reward
    global last_save_running_reward

    for now_epi in range(1, episode + 1):
        env.reset()

        sum_reward = 0

        print("OOOOOOO")

        for now_step in range(0, game_step):
            if env.done == True:
                break

            env.try_set_next_block()

            possible_state_with_reward = get_all_possible_state(env)

            V_with_possible_state = []
            for state in possible_state_with_reward:
                with torch.no_grad():
                    x = state_tensor(state[0].get_normalized_map())
                    x2 = torch.from_numpy(env.get_next_other_state_features()).to(
                        device=device, dtype=torch.float32
                    )
                    output = network(x, x2)
                    V_with_possible_state.append((output, state))

            # 寻找output最大的
            max_state_V = V_with_possible_state[0][0]
            max_state_p = 0
            for i in range(1, len(V_with_possible_state)):
                if max_state_V < V_with_possible_state[i][0]:
                    max_state_V = V_with_possible_state[i][0]
                    max_state_p = i
            ################################

            action = max_state_p

            func_render_game(possible_state_with_reward[action][0])

            sum_reward += possible_state_with_reward[action][1]
            if possible_state_with_reward[action][1] > 0:
                print(f"OHHHHH! network expect V:{V_with_possible_state[action][0]}")

            env.clone_from(possible_state_with_reward[action][0])

            time.sleep(0.05)


play()
