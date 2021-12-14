import numpy as np

x = [1, 2]
y = [1, 2]
t = [2, 2]
p = [[3, 2], [3, 1]]
p = np.array(p)
for a, b, c, d in zip(x, y, t, p):
    d += 1
print(p)


class aiueo:
    def __init__(self):
        self.n = np.zeros((2, 2))
        self.tt = 9


x = aiueo()
print(x.n.shape)

for a, b, c, d in zip([x.n, x.tt], y, t, p):
    a += 1

print(x.n)
print(x.tt)

c = np.zeros((1000, 1000))
import time

start = time.time()

x = []
