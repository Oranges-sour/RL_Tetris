from tetris_1 import Tetris
from tetris_1 import get_all_possible_state

import time

import pygame

import random

WW = Tetris.W - 4
HH = Tetris.H - 3

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((800, 600))
screen.fill((156, 156, 156))
pygame.display.set_caption("main")

env = Tetris()
env.reset()


for _ in range(0, 19):
    if env.done == True:
        break

    env.try_set_next_block()

    li = get_all_possible_state(env)
    kk = random.randint(0, len(li) - 1)
    env = li[kk][0]
    for i in li:
        time.sleep(0.017)

        for event in pygame.event.get():
            # 判断用户是否点了关闭按钮
            if event.type == pygame.QUIT:
                # 卸载所有模块
                pygame.quit()

        screen.fill((0, 0, 0))
        rrr = 17
        ss = i[0].get_colored_map()

        li_color = [
            (100, 240, 200),
            (120, 220, 100),
            (140, 200, 200),
            (160, 180, 100),
            (180, 160, 200),
            (200, 140, 100),
            (220, 120, 200),
        ]
        for x in range(0, HH):
            for y in range(0, WW):
                if ss[x][y] == 0:
                    continue
                pygame.draw.circle(
                    screen,
                    li_color[int(ss[x][y])],
                    (40 + y * 35, x * 35),
                    rrr,
                )
        pygame.display.flip()
