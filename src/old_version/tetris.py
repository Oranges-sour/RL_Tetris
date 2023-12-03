import random
from copy import deepcopy
from copy import copy
import numpy as np

W = 14
H = 21

bblock_que = [
    1,
    3,
    1,
    4,
    1,
    3,
    3,
    5,
    6,
    5,
    7,
    7,
    3,
    1,
    3,
    5,
    7,
    1,
    6,
    7,
    4,
    7,
    2,
    2,
    7,
    2,
    3,
    3,
    5,
    2,
    1,
    5,
    3,
    3,
    5,
    4,
    7,
    3,
    1,
    6,
    1,
    3,
    4,
    6,
    6,
    6,
    2,
    3,
    6,
    5,
    1,
    6,
    3,
    4,
    5,
    5,
    5,
    7,
    7,
    3,
    2,
    5,
    5,
    4,
    6,
    2,
    4,
    6,
    3,
    1,
    6,
    4,
    1,
    2,
    6,
    1,
    5,
    1,
    6,
    2,
    3,
    3,
    7,
    3,
    6,
    1,
    5,
    3,
    6,
    7,
    3,
    5,
    6,
    5,
    3,
    2,
    1,
    5,
    4,
    3,
    4,
    7,
    3,
    4,
    5,
    4,
    2,
    4,
    2,
    2,
    2,
    1,
    4,
    2,
    2,
    2,
    1,
    4,
    4,
    5,
    3,
    6,
    1,
    6,
    1,
    6,
    5,
    1,
    3,
    3,
    4,
    6,
    7,
    6,
    5,
    4,
    5,
    5,
    4,
    6,
    6,
    5,
    3,
    1,
    2,
    4,
    3,
    1,
    5,
    6,
    6,
    2,
    4,
    1,
    4,
    1,
    5,
    6,
    4,
    3,
    1,
    3,
    7,
    4,
    6,
    4,
    6,
    7,
    1,
    1,
    1,
    6,
    6,
    4,
    4,
    6,
    4,
    2,
    2,
    6,
    2,
    3,
    7,
    5,
    7,
    7,
    2,
    1,
    2,
    5,
    3,
    5,
    2,
    3,
    4,
    3,
    4,
    2,
    5,
    7,
    3,
    3,
    4,
    4,
    5,
    5,
    4,
    5,
    6,
    5,
    5,
    5,
    7,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    3,
    7,
    3,
    5,
    4,
    4,
    1,
    3,
    4,
    4,
    5,
    7,
    7,
    1,
    4,
    3,
    1,
    1,
    6,
    7,
    1,
    5,
    5,
    7,
    4,
    4,
    1,
    7,
    4,
    7,
    6,
    7,
    6,
    6,
    3,
    3,
    7,
    2,
    3,
    5,
    1,
    7,
    4,
    4,
    3,
    3,
    6,
    6,
    4,
    3,
    5,
    6,
    4,
    1,
    7,
    4,
    7,
    6,
    1,
    4,
    4,
    5,
    2,
    7,
    4,
    3,
    3,
    2,
    4,
    7,
    3,
    4,
    6,
    2,
    4,
    7,
    5,
    6,
    5,
    2,
    6,
    4,
    6,
    2,
    5,
    2,
    6,
    4,
    3,
    6,
    1,
    3,
    1,
    6,
    4,
    7,
    4,
    5,
    5,
    4,
    7,
    1,
    4,
    5,
    4,
    3,
    3,
    2,
    1,
    6,
    5,
    1,
    3,
    6,
    5,
    4,
    2,
    5,
    7,
    4,
    5,
    6,
    4,
    1,
    4,
    6,
    2,
    3,
    3,
    5,
    6,
    5,
    1,
    1,
    4,
    5,
    4,
    4,
    4,
]

ccolor_que = []
for _ in range(0, 180):
    k = random.randint(1, 5)
    ccolor_que.append(k)


# 是否是不移动的颜色块
def is_scolor_block(k, C):
    if k != 0 and k != C and k != 7:
        return True
    return False


# def get_all_possible_state(now_state):


# def dfs_state(last_state,action):


class Tetris:
    W = W
    H = H

    def __init__(self) -> None:
        return

    def clone(self):
        new_t = Tetris()

        new_t.map = deepcopy(self.map)

        new_t.game_point = self.game_point
        new_t.game_point_temp = self.game_point_temp
        new_t.can_next_block = self.can_next_block
        new_t.move_R = self.move_R
        new_t.block = self.block
        new_t.C = self.C
        new_t.done = self.done
        new_t.block_que = copy(self.block_que)
        new_t.color_que = copy(self.color_que)

        return new_t

    def clone_from(self, other_t):
        for i in range(0, H):
            for j in range(0, W):
                self.map[i][j] = other_t.map[i][j]

        self.game_point = other_t.game_point
        self.game_point_temp = other_t.game_point_temp
        self.can_next_block = other_t.can_next_block
        self.move_R = other_t.move_R
        self.block = other_t.block
        self.C = other_t.C
        self.done = other_t.done
        self.block_que = copy(other_t.block_que)
        self.color_que = copy(other_t.color_que)

    def reset(self):
        # 游戏地图
        self.map = [
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        ]

        self.game_point = 0  # 游戏分数
        self.game_point_temp = 0

        self.can_next_block = True

        self.move_R = 0

        self.block = 0

        self.C = 0

        self.done = False

        self.count = 0

        self.block_que = copy(bblock_que)

        self.color_que = copy(ccolor_que)

    def get_normalized_map(self):
        normalized_map = np.zeros((H - 3, W - 4))
        # print(normalized_map.shape)
        for x in range(0, H - 3):
            for y in range(0, W - 4):
                if self.map[x][y + 2] == 0:
                    normalized_map[x][y] = 0

                if (
                    self.map[x][y + 2] != 0
                    and self.map[x][y + 2] != 7
                    and self.map[x][y + 2] != self.C
                ):
                    normalized_map[x][y] = 0.5
                if self.map[x][y + 2] == self.C:
                    normalized_map[x][y] = 1
        # print(normalized_map)
        return normalized_map

    def get_colored_map(self):
        colored_map = np.zeros((H - 3, W - 4))
        for x in range(0, H - 3):
            for y in range(0, W - 4):
                if self.map[x][y + 2] >= 8:
                    colored_map[x][y] = self.map[x][y + 2] - 7
                else:
                    colored_map[x][y] = self.map[x][y + 2]

        return colored_map

    def step(self, action):
        if self.done:
            return (self.get_normalized_map(), 0, True)

        self.count += 1

        self.check_done()
        self.set_block()
        if self.count % 2 == 0:
            self.auto_move()
        self.clear()

        # if action == 0:
        #     _
        if action == 1:
            self.left()
        if action == 2:
            self.right()
        if action == 3:
            self.turn()

        reward = 0
        if self.game_point_temp != 0:
            print("OHHHHH!")
            rrr = 0
            for i in range(1, self.game_point_temp + 1):
                rrr += 1
            self.game_point_temp = 0
            reward += rrr * 10

        return (self.get_normalized_map(), reward, False)

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ############################      游戏实现部分     ##############################
    ###############################################################################

    def check_done(self):
        if self.game_point > 10:
            self.done = True
            return
        for y in range(0, W):
            if is_scolor_block(self.map[5][y], self.C):
                self.done = True
                break

    def auto_move(self):
        move = True
        for x in range(H - 2, -1, -1):
            for y in range(0, W):
                if (
                    self.map[x][y] == self.C
                    and self.map[x + 1][y] != 0
                    and self.map[x + 1][y] != self.C
                ):
                    move = False

        if move == False:
            for x in range(0, H):
                for y in range(0, W):
                    if self.map[x][y] == 1:
                        self.map[x][y] = 8

                    if self.map[x][y] == 2:
                        self.map[x][y] = 9

                    if self.map[x][y] == 3:
                        self.map[x][y] = 10

                    if self.map[x][y] == 4:
                        self.map[x][y] = 11

                    if self.map[x][y] == 5:
                        self.map[x][y] = 12
            self.can_next_block = True
            return

        for x in range(H - 2, -1, -1):
            for y in range(0, W):
                if self.map[x][y] == self.C:
                    self.map[x + 1][y] = self.C
                    self.map[x][y] = 0

    def set_block(self):
        if self.can_next_block == False:
            return

        self.move_R = 0
        self.random_block()
        self.random_color()

        if self.block == 1:
            self.map[1][3] = self.C
            self.map[1][4] = self.C
            self.map[1][5] = self.C
            self.map[1][6] = self.C
        if self.block == 2:
            self.map[0][4] = self.C
            self.map[1][4] = self.C
            self.map[2][4] = self.C
            self.map[2][5] = self.C
        if self.block == 3:
            self.map[0][4] = self.C
            self.map[1][4] = self.C
            self.map[2][4] = self.C
            self.map[2][3] = self.C
        if self.block == 4:
            self.map[0][4] = self.C
            self.map[1][3] = self.C
            self.map[1][4] = self.C
            self.map[1][5] = self.C
        if self.block == 5:
            self.map[0][4] = self.C
            self.map[1][4] = self.C
            self.map[1][5] = self.C
            self.map[2][5] = self.C
        if self.block == 6:
            self.map[0][4] = self.C
            self.map[1][4] = self.C
            self.map[1][3] = self.C
            self.map[2][3] = self.C
        if self.block == 7:
            self.map[0][4] = self.C
            self.map[0][3] = self.C
            self.map[1][4] = self.C
            self.map[1][3] = self.C

        # print(self.get_normalized_map())

        self.can_next_block = False

    def clear(self):
        x = H - 1
        while x >= 0:
            clear_count = 0
            for y in range(0, W):
                if (
                    self.map[x][y] == 8
                    or self.map[x][y] == 9
                    or self.map[x][y] == 10
                    or self.map[x][y] == 11
                    or self.map[x][y] == 12
                ):
                    clear_count += 1

                if clear_count >= W - 4:  # 一行满了,允许x个空位
                    self.game_point += 1
                    self.game_point_temp += 1

                    for j in range(0, W):
                        if self.map[x][j] != 7:
                            self.map[x][j] = 0

                    self.clear_Move(x)

                    x = H - 1
                    break
            x -= 1

    def clear_Move(self, clear_X):
        for x in range(clear_X, -1, -1):
            for y in range(0, W):
                if (
                    self.map[x][y] == 8
                    or self.map[x][y] == 9
                    or self.map[x][y] == 10
                    or self.map[x][y] == 11
                    or self.map[x][y] == 12
                ) and self.map[x + 1][y] == 0:
                    if self.map[x][y] == 8:
                        self.map[x][y] = 0
                        self.map[x + 1][y] = 8
                    if self.map[x][y] == 9:
                        self.map[x][y] = 0
                        self.map[x + 1][y] = 9

                    if self.map[x][y] == 10:
                        self.map[x][y] = 0
                        self.map[x + 1][y] = 10

                    if self.map[x][y] == 11:
                        self.map[x][y] = 0
                        self.map[x + 1][y] = 11

                    if self.map[x][y] == 12:
                        self.map[x][y] = 0
                        self.map[x + 1][y] = 12

    def turn(self):
        if self.block == 7:
            return

        move_X = int(-1)
        move_Y = int(-1)

        for x in range(0, H):
            for y in range(0, W):
                if self.map[x][y] == self.C:
                    move_X = x
                    move_Y = y

        if move_X == -1 or move_Y == -1:
            return

        temp_map = np.zeros((H, W))

        for x in range(0, H):
            for y in range(0, W):
                temp_map[x][y] = self.map[x][y]

        block_count_1 = 0
        for x in range(0, H):
            for y in range(0, W):
                if self.map[x][y] != 0 and self.map[x][y] != self.C:
                    block_count_1 += 1

        for x in range(0, H):
            for y in range(0, W):
                if self.map[x][y] == self.C:
                    self.map[x][y] = 0

        if self.move_R == 0:
            self.turn_0(move_X, move_Y)
            self.move_R += 1

        elif self.move_R == 1:
            self.turn_1(move_X, move_Y)
            self.move_R += 1

        elif self.move_R == 2:
            self.turn_2(move_X, move_Y)
            self.move_R += 1

        elif self.move_R == 3:
            self.turn_3(move_X, move_Y)
            self.move_R = 0

        block_count_2 = 0
        for x in range(0, H):
            for y in range(0, W):
                if self.map[x][y] != 0 and self.map[x][y] != self.C:
                    block_count_2 += 1

        if block_count_1 != block_count_2:
            for x in range(0, H):
                for y in range(0, W):
                    self.map[x][y] = temp_map[x][y]

    def right(self):
        jumpout = False

        move = True
        for y in range(W - 1, -1, -1):
            for x in range(0, H):
                if (
                    self.map[x][y] == self.C
                    and self.map[x][y + 1] != 0
                    and self.map[x][y + 1] != self.C
                ):
                    move = False
                    jumpout = True
                    break

            if jumpout == True:
                break

        if move == True:
            for y in range(W - 1, -1, -1):
                for x in range(0, H):
                    if self.map[x][y] == self.C:
                        self.map[x][y] = 0
                        self.map[x][y + 1] = self.C

    def left(self):
        jumpout = False

        move = True
        for y in range(1, W):
            for x in range(0, H):
                if (
                    self.map[x][y] == self.C
                    and self.map[x][y - 1] != 0
                    and self.map[x][y - 1] != self.C
                ):
                    move = False
                    jumpout = True
                    break

            if jumpout == True:
                break

        if move == True:
            for y in range(1, W):
                for x in range(0, H):
                    if self.map[x][y] == self.C:
                        self.map[x][y] = 0
                        self.map[x][y - 1] = self.C

    def random_color(self):
        self.C = self.color_que[len(self.color_que) - 1]
        self.color_que.pop(len(self.color_que) - 1)

    def random_block(self):
        # self.block = 1
        self.block = self.block_que[len(self.block_que) - 1]
        self.block_que.pop(len(self.block_que) - 1)

    def turn_0(self, move_X, move_Y):
        if self.block == 1:
            self.map[move_X - 1][move_Y - 1] = self.C
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C

        if self.block == 2:
            self.map[move_X - 1][move_Y - 2] = self.C
            self.map[move_X - 1][move_Y - 1] = self.C
            self.map[move_X - 1][move_Y] = self.C
            self.map[move_X][move_Y - 2] = self.C

        if self.block == 3:
            self.map[move_X - 1][move_Y - 2] = self.C
            self.map[move_X][move_Y - 2] = self.C
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C

        if self.block == 4:
            self.map[move_X - 1][move_Y - 1] = self.C
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C

        if self.block == 5:
            self.map[move_X - 1][move_Y - 1] = self.C
            self.map[move_X - 1][move_Y] = self.C
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y - 2] = self.C

        if self.block == 6:
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X - 1][move_Y - 1] = self.C
            self.map[move_X - 1][move_Y] = self.C

    def turn_1(self, move_X, move_Y):
        if self.block == 1:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X][move_Y + 2] = self.C

        if self.block == 2:
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X + 1][move_Y + 1] = self.C
            self.map[move_X + 2][move_Y + 1] = self.C

        if self.block == 3:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C

        if self.block == 4:
            self.map[move_X - 1][move_Y - 1] = self.C
            self.map[move_X - 1][move_Y] = self.C
            self.map[move_X - 1][move_Y + 1] = self.C
            self.map[move_X][move_Y] = self.C

        if self.block == 5:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 2][move_Y] = self.C

        if self.block == 6:
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C

    def turn_2(self, move_X, move_Y):
        if self.block == 1:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C
            self.map[move_X + 3][move_Y - 1] = self.C

        if self.block == 2:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X - 1][move_Y + 1] = self.C

        if self.block == 3:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X + 1][move_Y + 1] = self.C

        if self.block == 4:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X - 1][move_Y] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y] = self.C

        if self.block == 5:
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y] = self.C

        if self.block == 6:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 1][move_Y + 1] = self.C

    def turn_3(self, move_X, move_Y):
        if self.block == 1:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X][move_Y] = self.C
            self.map[move_X][move_Y + 1] = self.C
            self.map[move_X][move_Y + 2] = self.C

        if self.block == 2:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C
            self.map[move_X + 2][move_Y] = self.C

        if self.block == 3:
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 2][move_Y] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C

        if self.block == 4:
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 1][move_Y + 1] = self.C

        if self.block == 5:
            self.map[move_X][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 2][move_Y] = self.C

        if self.block == 6:
            self.map[move_X][move_Y] = self.C
            self.map[move_X + 1][move_Y - 1] = self.C
            self.map[move_X + 1][move_Y] = self.C
            self.map[move_X + 2][move_Y - 1] = self.C
