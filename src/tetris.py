import random
from copy import deepcopy
from copy import copy
import numpy as np

W = 14
H = 21

bblock_que = []
ccolor_que = []


def random_block_color_que(seed):
    global bblock_que
    global ccolor_que

    # 固定序列随机
    rng = random.Random(seed)

    bblock_que.clear()
    for _ in range(0, 1000):
        bblock_que.append(rng.randint(1, 7))

    ccolor_que.clear()
    for _ in range(0, 1000):
        k = rng.randint(1, 5)
        ccolor_que.append(k)


random_block_color_que(114514)


# 是否是不移动的颜色块
def is_scolor_block(k, C):
    if k != 0 and k != C and k != 7:
        return True
    return False


class Tetris:
    W = W
    H = H

    def __init__(self, reward_per_line) -> None:
        self.reward_per_line = reward_per_line
        return

    def clone(self):
        new_t = Tetris(self.reward_per_line)

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

        self.reward_per_line = other_t.reward_per_line
        self.game_point = other_t.game_point
        self.game_point_temp = other_t.game_point_temp
        self.can_next_block = other_t.can_next_block
        self.move_R = other_t.move_R
        self.block = other_t.block
        self.C = other_t.C
        self.done = other_t.done
        self.block_que = copy(other_t.block_que)
        self.color_que = copy(other_t.color_que)

    def reset(self, random_bc_que=False, random_bc_seed=0):
        # 游戏地图
        self.map = [
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 0
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 1
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 2
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 3
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 4
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 5
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 6
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 7
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 8
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 9
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 10
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 11
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 12
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 13
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 14
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 15
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 16
            [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7],  # 17
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

        if random_bc_que:
            random_block_color_que(random_bc_seed)

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
                    normalized_map[x][y] = 1
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

    def get_next_other_state_features(self):
        next = np.zeros(7 * 3 + 10 + 10)
        cnt = 0
        # 即将来的3个块的one-hot
        for i in range(len(self.block_que) - 1, len(self.block_que) - 4, -1):
            next[self.block_que[i] - 1 + cnt * 7] = 1
            cnt += 1
        # print("____________________________________________")
        # for x in range(0, H - 3):
        #     for y in range(0, W - 4):
        #         print(f"{self.map[x][y + 2]:<3}", end="")
        #     print("")
        temp = np.zeros(10)
        # 每一列最高块的高度
        for i in range(0, W - 4):
            count = H - 3
            for j in range(0, H - 3):
                if is_scolor_block(self.map[j][i + 2], self.C):
                    count = j
                    break
            temp[i] = count
            next[7 * 3 + i] = (H - 3 + 1) - count - 1
        # 每一列的空洞数量
        for i in range(0, W - 4):
            count = 0
            for j in range(int(temp[i]), H - 3):
                if self.map[j][i + 2] == 0:
                    count += 1
            next[7 * 3 + 10 + i] = count
        # print(next)
        return next

    def try_set_next_block(self):
        if self.can_next_block:
            self.set_block()

    def step(self, type, param1, param2):
        if self.done:
            return 0

        if type == 1:
            for i in range(0, param1):
                self.turn()
        if type == 2:
            if param1 == 1:
                for _ in range(0, param2):
                    self.left()
            if param1 == 2:
                for _ in range(0, param2):
                    self.right()

        if type == 3:
            while not self.can_next_block:
                self.auto_move()

        self.clear()

        reward = 0
        if self.game_point_temp != 0:
            # print("OHHHHH!")
            rrr = 0
            for i in range(1, self.game_point_temp + 1):
                rrr += 1
            self.game_point_temp = 0
            reward += rrr * self.reward_per_line
        # reward += 0.1
        # print(self.reward_per_line)

        self.check_done()

        return reward

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ############################      游戏实现部分     ##############################
    ###############################################################################

    def check_done(self):
        if self.game_point > 180:
            self.done = True
            return
        for y in range(0, W):
            if is_scolor_block(self.map[1][y], self.C):
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


def get_map_hash(map):
    p = int(1e9 + 7)

    ss = 0
    for i in range(0, H - 3):
        for j in range(0, W - 4):
            ss = (ss * 131 + int(map[i][j] * 2) * 10159) % p

    return ss


class StatePool:
    def __init__(self, n) -> None:
        self.state_pool = []
        self.cnt = 0

        for _ in range(0, n):
            tt = Tetris(1)
            tt.reset()
            self.state_pool.append(tt)

    def get(self):
        k = self.state_pool[self.cnt]
        self.cnt += 1
        # print(self.cnt)
        return k

    def reset(self):
        self.cnt = 0


state_pool = StatePool(1000)


def get_all_possible_state(now_state: Tetris, debug=False):
    global state_pool

    state_pool.reset()

    dd = dict()
    dfs_state(now_state, dd, 1, 0, debug)

    li = []
    for v in dd.values():
        li.append(v)
    return li


def dfs_state(last_state: Tetris, dd: dict, depth, reward, debug=False):
    global state_pool

    if depth == 1:
        for i in range(0, 4):
            k = state_pool.get()
            k.clone_from(last_state)

            m = k.step(depth, i, 0)

            dfs_state(k, dd, depth + 1, m + reward)

    if depth == 2:
        for i in range(0, 8):
            k = state_pool.get()
            k.clone_from(last_state)
            # k = last_state.clone()
            m = k.step(depth, 1, i)

            dfs_state(k, dd, depth + 1, m + reward)
        for i in range(0, 8):
            k = state_pool.get()
            k.clone_from(last_state)
            # k = last_state.clone()
            m = k.step(depth, 2, i)

            dfs_state(k, dd, depth + 1, m + reward)

    if depth == 3:
        k = state_pool.get()
        k.clone_from(last_state)
        # k = last_state.clone()
        m = k.step(depth, 0, 0)

        hs = get_map_hash(k.get_normalized_map())

        if not hs in dd:
            dd[hs] = (k, m + reward)
