from sum_tree import SumTree
import numpy as np
import random
import math

print("SumTree Test")

li = np.zeros(30005)



tree = SumTree(30000)

for i in range(1, 30000):
    tree.insert(i, 1)

tree.insert(1, 100)

for _ in range(int(1e6)):
    k = random.randint(1, tree.get_sum())

    jj = tree.find(k)
    li[jj] += 1


print("OOO")

li = np.sort(li)

for i in range(len(li) - 1, len(li) - 10, -1):
    print(li[i])
