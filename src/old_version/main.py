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


from old_version.tetris import Tetris
from sum_tree import SumTree


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

from torch.utils.tensorboard.writer import SummaryWriter

# 日志目录
log_dir_num = f"{int(time.time())}"

os.system(f"mkdir logs/{log_dir_num}")


writer = SummaryWriter(f"logs/{log_dir_num}")

print(f"logs/{log_dir_num}")

# 开始时的时间，用来记录训练了多久
time0 = time.time()

# cpu训练
device = "cpu"

# 游戏环境
env = Tetris()



# 动作空间
action_n = 4
# 观察空间
obser_n = WW * HH


render_game = False


# Dueling Q Network
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()

        self.out_channel1 = 32

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channel1,
            kernel_size=3,
            padding=1,
            device=device,
        )
        self.batchnorm1 = nn.BatchNorm2d(self.out_channel1, device=device)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            self.out_channel1 * int(WW / 2) * int(HH / 2), 128, device=device
        )

        self.fc2 = nn.Linear(128, 64, device=device)

        self.fc2_v = nn.Linear(64, 1, device=device)

        self.fc2_a = nn.Linear(64, action_n, device=device)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 1, HH, WW)

        x = self.activation(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool1(x)

        x = x.reshape(-1, self.out_channel1 * int(WW / 2) * int(HH / 2))

        x = self.activation(self.fc1(x))

        x = self.activation(self.fc2(x))

        V = self.fc2_v(x)
        A = self.fc2_a(x)

        A = A - A.max()

        Q = V + A

        return Q


# 双网络交替
network = Network()

# 使用上次效果好的继续训练
# network = torch.load("model/1699079142.pth")

target_network = Network()
target_network.load_state_dict(network.state_dict())


# 游戏次数
episode = 2500

# 一些超参数

batch_size = 32

learning_rate = 0.0003

# 折扣率
gamma = 0.9995

# 探索率
e = 0.1
# 探索率衰减值
e_decay = 0.9991

# 一个episode的游戏最大步数
game_step = 3000
###########################

# 获得的reward，动态更新
running_reward = None

last_save_running_reward = 0

# 所有episode的游戏步数加和
game_frame_count = 0

# optimizer = optim.SGD(network.parameters(), lr=0.03, weight_decay=0)
optimizer = optim.Adam(
    network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08
)
correction = nn.MSELoss()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity

        self.writer = 1

        self.tree = SumTree(capacity)
        self.buffer = [[None, None, None, None, None]]

    def push(self, s, action, reward, new_s, is_end, priority=1):
        kk = [state_tensor(s), action, reward, state_tensor(new_s), is_end]

        if len(self.buffer) >= self.capacity:
            self.buffer[self.writer] = kk
            self.tree.insert(self.writer, priority)

            self.writer += 1
            if self.writer >= self.capacity:
                self.writer = 1
            return

        self.buffer.append(kk)
        self.tree.insert(len(self.buffer) - 1, priority)

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


replay = ReplayBuffer(30000)


def state_tensor(s):
    x = torch.from_numpy(s).to(device=device, dtype=torch.float32)
    return x


def to_tensor(kk):
    t = torch.zeros((batch_size, HH, WW), device=device)
    for i in range(0, batch_size):
        t[i] = kk[i]
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


for now_episode in range(0, episode):
    if now_episode % 10 == 0:
        target_network.load_state_dict(network.state_dict())

    env.reset()

    sum_reward = 0

    lenl = 1

    for _ in range(0, game_step):
        if env.done == True:
            break
        func_render_game(env)

        game_frame_count += 1

        aa = 0

        old_s = env.get_normalized_map()

        x = state_tensor(old_s)
        output = network(x)
        aa = output.argmax().item()

        # 随机动作选择
        k = random.uniform(0, 1)
        if k < e:
            aa = random.randint(0, action_n - 1)

        result = env.step(aa)
        new_s = result[0]
        reward = result[1]
        is_end = result[2]

        sum_reward += reward

        replay.push(
            old_s,
            aa,
            reward,
            new_s,
            float(is_end),
             max(1, int(0.5 * math.pow((output.argmax() - reward), 2))),
            #1,
        )

        trans = replay.sample(batch_size)

        # batch的大小足够了（不是None），开始训练
        if trans != None:
            batch = list(zip(*trans))

            s_batch = to_tensor(batch[0])
            new_s_batch = to_tensor(batch[3])

            aa_batch = torch.tensor(batch[1], dtype=torch.int32, device=device)
            reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)

            done_batch = torch.tensor(batch[4], dtype=torch.float32, device=device)

            q_v = network(s_batch)
            with torch.no_grad():
                new_q_v = target_network(new_s_batch)
            new_q_v = new_q_v.max(1)[0]

            expect_q_v = reward_batch + gamma * new_q_v * (1 - done_batch)

            target_q_v = q_v.detach().clone()
            for i in range(0, batch_size):
                target_q_v[i][aa_batch[i]] = expect_q_v[i]

            # print("oioi")
            # print(q_v)
            # print(target_q_v)

            loss = correction(q_v, target_q_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), now_episode)

    writer.add_scalar("reward", sum_reward, now_episode)

    if running_reward == None:
        running_reward = sum_reward
    else:
        e = max(e * e_decay, 0.001)
        running_reward = running_reward * 0.9 + sum_reward * 0.1

    if running_reward > 10 and running_reward > last_save_running_reward:
        last_save_running_reward = running_reward
        print(f"save model: rewawd:{running_reward}")
        torch.save(network, f"model/{log_dir_num}.pth")

    if now_episode % lenl == 0:
        print(
            f"epoch={now_episode}:epsilon:{e:.4f}  sum_reward:{sum_reward:.4f} running_reward:{running_reward:.8f}  total_time:{time.time() - time0:.2f}s"
        )


# env.close()
