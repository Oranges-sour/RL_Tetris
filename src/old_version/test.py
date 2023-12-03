import pygame
import time

from old_version.tetris import Tetris

env = Tetris()

# 使用pygame之前必须初始化
pygame.init()
# 设置主屏窗口
screen = pygame.display.set_mode((600, 800))
screen.fill((0, 0, 0))
pygame.display.set_caption("test")

s = env.reset()[0]

# print(s)

while True:
    action = 0
    for event in pygame.event.get():
        # 判断用户是否点了关闭按钮
        if event.type == pygame.QUIT:
            # 卸载所有模块
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                action = 3
            if event.key == pygame.K_a:
                action = 1
            if event.key == pygame.K_d:
                action = 2

    screen.fill((0, 0, 0))
    rrr = 17
    for x in range(0, Tetris.H - 3):
        for y in range(0, Tetris.W - 4):
            if s[x][y] == 1:
                pygame.draw.circle(screen, (100, 150, 50), (40 + y * 35, x * 35), rrr)

            if s[x][y] == 2:
                pygame.draw.circle(screen, (180, 150, 120), (40 + y * 35, x * 35), rrr)
            if s[x][y] == 3:
                pygame.draw.circle(screen, (100, 120, 150), (40 + y * 35, x * 35), rrr)

    pygame.display.flip()

    time.sleep(0.15)
    s = env.step(action)[0]
