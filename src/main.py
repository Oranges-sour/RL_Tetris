import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np

import copy

import time
import os

import pygame

import math


from tetris import Tetris
from tetris import get_all_possible_state

from sum_tree import SumTree

from main_network import Network


from torch.utils.tensorboard.writer import SummaryWriter


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity

        self.writer = 1

        self.tree = SumTree(capacity)
        self.buffer = [[None, None, None, None, None]]

    def push(self, s, next_3_block, reward, new_s, is_end, priority=1):
        kk = [
            state_tensor(s),
            next_3_block,
            reward,
            state_tensor(new_s),
            is_end,
        ]

        if len(self.buffer) >= self.capacity:
            self.buffer[self.writer] = kk
            self.tree.insert(self.writer, priority)

            self.writer += 1

            result = True

            if self.writer >= self.capacity:
                self.writer = 1
                result = False

            return result

        self.buffer.append(kk)
        self.tree.insert(len(self.buffer) - 1, priority)

        return True

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        li = []
        sum = self.tree.get_sum()
        for _ in range(0, batch_size):
            k = random.randint(1, sum)
            j = self.tree.find(k)
            li.append(self.buffer[j])

        return li


WW = Tetris.W - 4
HH = Tetris.H - 3

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((WW * 20, HH * 20))
pygame.display.set_caption("main")

img = []
for i in range(1, 6):
    img.append(
        pygame.transform.scale(pygame.image.load(f"blocks/{i}.jpg"), (20, 20)).convert()
    )


# cpu训练
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# 观察空间
obser_n = WW * HH

render_game = False


# ###########################


def state_tensor(s):
    x = torch.from_numpy(s).to(device=device, dtype=torch.float32)
    return x


def to_tensor_1(kk, batch_size):
    t = torch.zeros((batch_size, HH, WW), device=device)
    for i in range(0, batch_size):
        t[i] = kk[i]
    return t


def to_tensor_2(kk, batch_size):
    t = torch.zeros((batch_size, 3 * 7 + 10 + 10), device=device)

    for i in range(0, batch_size):
        p = torch.from_numpy(kk[i]).to(device=device, dtype=torch.float32)
        t[i] = p
    return t


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
    pass


def train(
    episode,
    epsilon_func,
    gamma,
    lr,
    reward_per_line,
    replay_buffer_size,
    batch_size,
    score_k,
):
    print(f"<<train start>>")
    print(f"episode:{episode},")

    print(f"gamma:{gamma},")
    print(f"lr:{lr},")
    print(f"reward_per_line:{reward_per_line},")
    print(f"replay_buffer_size:{replay_buffer_size},")
    print(f"batch_size:{batch_size}")
    print(f"score_k:{score_k}")

    print(f"<<train start>>")

    # 日志目录
    log_dir_num = f"{int(time.time())}"
    print(f"logs/{log_dir_num}")

    env = Tetris(reward_per_line)

    # 一个episode的游戏最大步数
    game_step = 3000

    # 双网络交替
    network = Network()
    #在现有模型上继续训练
    network = torch.load("model/1701667197_final.pth", map_location=device)
    target_network = Network()
    target_network.load_state_dict(network.state_dict())

    # optimizer = optim.SGD(network.parameters(), lr=lr, weight_decay=0.000003)
    optimizer = optim.Adam(
        network.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    correction = nn.MSELoss()

    replay = ReplayBuffer(replay_buffer_size)

    writer = SummaryWriter(f"logs/{log_dir_num}")

    # 开始时的时间，用来记录训练了多久
    time0 = time.time()

    # 获得的reward，动态更新
    running_reward = None

    last_save_running_reward = 0

    for now_episode in range(1, episode + 1):
        epsilon = epsilon_func(now_episode)

        if now_episode % 10 == 0:
            target_network.load_state_dict(network.state_dict())

        env.reset()

        sum_reward = 0

        lenl = 1

        for _ in range(0, game_step):
            if env.done == True:
                break

            env.try_set_next_block()

            possible_state_with_reward = get_all_possible_state(env)

            V_with_possible_state = []

            network.eval()

            for state in possible_state_with_reward:
                with torch.no_grad():
                    x = state_tensor(state[0].get_normalized_map())
                    x2 = torch.from_numpy(env.get_next_other_state_features()).to(
                        device=device, dtype=torch.float32
                    )
                    output = network(x, x2)
                    V_with_possible_state.append((output, state))

                result = replay.push(
                    env.get_normalized_map(),
                    env.get_next_other_state_features(),
                    state[1],
                    state[0].get_normalized_map(),
                    state[0].done,
                    # int(
                    #     max(
                    #         1,
                    #         score_k
                    #         * math.log10(
                    #             math.pow(
                    #                 output - state[1],
                    #                 2,
                    #             )
                    #             + 1
                    #         ),
                    #     )
                    # ),
                    1,
                )

                if result == False:
                    print(f"replay buffer full once, episode:{now_episode}")
            network.train()

            # 寻找output最大的
            max_state_V = V_with_possible_state[0][0]
            max_state_p = 0
            for i in range(1, len(V_with_possible_state)):
                if max_state_V < V_with_possible_state[i][0]:
                    max_state_V = V_with_possible_state[i][0]
                    max_state_p = i
            ################################

            action = max_state_p
            # 随机动作选择
            k = random.uniform(0, 1)
            if k < epsilon:
                action = random.randint(0, len(V_with_possible_state) - 1)

            func_render_game(possible_state_with_reward[action][0])

            sum_reward += possible_state_with_reward[action][1] / reward_per_line

            env.clone_from(possible_state_with_reward[action][0])

            ####### train ########
            train_step = 1

            for _ in range(0, train_step):
                trans = replay.sample(batch_size)
                # batch的大小足够了（不是None），开始训练
                if trans != None:
                    batch = list(zip(*trans))

                    s_batch = to_tensor_1(batch[0], batch_size)
                    next_3_block = to_tensor_2(batch[1], batch_size)
                    new_s_batch = to_tensor_1(batch[3], batch_size)
                    reward_batch = torch.tensor(
                        batch[2], dtype=torch.float32, device=device
                    )

                    # reward_batch = (reward_batch - reward_batch.mean()) / (
                    #     reward_batch.std() + 1e-7
                    # )
                    done_batch = torch.tensor(
                        batch[4], dtype=torch.float32, device=device
                    )

                    q_v = network(s_batch, next_3_block)
                    # print(q_v)
                    with torch.no_grad():
                        new_q_v = target_network(new_s_batch, next_3_block)
                    new_q_v = new_q_v.reshape(-1)

                    expect_q_v = reward_batch + gamma * new_q_v * (1 - done_batch)

                    expect_q_v = expect_q_v.unsqueeze(1)

                    loss = correction(q_v, expect_q_v)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar("loss", loss.item(), now_episode)

        writer.add_scalar("reward", sum_reward, now_episode)
        writer.add_scalar("e", epsilon, now_episode)

        if running_reward == None:
            running_reward = sum_reward
        else:
            running_reward = running_reward * 0.9 + sum_reward * 0.1

        if running_reward > 10 and running_reward > last_save_running_reward:
            last_save_running_reward = running_reward
            print(f"save model: rewawd:{running_reward}")
            torch.save(network, f"model/{log_dir_num}_max_reward.pth")
        if now_episode % 500 == 0:
            torch.save(network, f"model/{log_dir_num}_{now_episode}.pth")

        # if now_episode % lenl == 0:
        #     print(
        #         f"epoch={now_episode}:epsilon:{epsilon:.4f}  sum_reward:{sum_reward:.4f} running_reward:{running_reward:.8f}  total_time:{time.time() - time0:.2f}s"
        #     )
        #     # logging.info(
        #     #     f"epoch={now_episode}:epsilon:{epsilon:.4f}  sum_reward:{sum_reward:.4f} running_reward:{running_reward:.8f}  total_time:{time.time() - time0:.2f}s"
        #     # )

    print(f"finish: total_time:{time.time() - time0:.2f}s")

    torch.save(network, f"model/{log_dir_num}_final.pth")
