import numpy as np

#基于线段树的加和树

def ls(x):
    return int(x * 2)


def rs(x):
    return int(x * 2 + 1)


class SumTree:
    def __init__(self, capacity) -> None:
        self.capacity = capacity

        self.tree = np.zeros((capacity * 4), dtype=int)

        self.build_tree(1, capacity, 1)

        return

    def build_tree(self, l, r, p):
        if l == r:
            self.tree[p] = 0
            return

        mid = int((l + r) / 2)
        self.build_tree(l, mid, ls(p))
        self.build_tree(mid + 1, r, rs(p))

    def get_sum(self):
        return self.tree[1]

    def insert(self, k, v):
        self.ins(1, self.capacity, 1, k, int(v))

    def ins(self, l, r, p, k, v):
        if k < l or k > r:
            return
        if l == r and l == k:
            self.tree[p] = v
            return

        mid = int((l + r) / 2)
        self.ins(l, mid, ls(p), k, v)
        self.ins(mid + 1, r, rs(p), k, v)

        self.tree[p] = self.tree[ls(p)] + self.tree[rs(p)]

    def find(self, v):
        return self.fin(1, self.capacity, 1, int(v))

    def fin(self, l, r, p, v):
        if l == r:
            return l
        #print(f"{l},{r}")
        result = None

        mid = int((l + r) / 2)
        # 先找左树
        if self.tree[ls(p)] >= v:
            result = self.fin(l, mid, ls(p), v)
        else:
            result = self.fin(mid + 1, r, rs(p), v - self.tree[ls(p)])
        return result

