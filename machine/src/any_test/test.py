import numpy as np


def normalize(o):
    mean = o.mean(axis=0)
    o1 = o - mean
    var = np.mean(o1 ** 2, axis=0)
    std = np.sqrt(var + 10e-7)
    return o1 / std


lr = 0.0012
e = 0
s = 0
for i in range(100):
    e += 1
    nn = 2
    x = np.random.randint(1, 5, (nn, 4, 3)).astype(np.float)
    t = np.random.randint(1, 100, nn)
    print("t:", t)
    w = np.random.randint(1, 10, (3, 1)).astype(np.float)
    one_w = np.ones((4, 1))
    for _ in range(100):
        n, i, nodes = x.shape
        xx = x.reshape(n * i, nodes)
        # else:
        #     xx = x
        ans = np.dot(xx, w)
        ans = ans.reshape((n, i))
        ans = np.dot(ans, one_w)
        ans = ans.reshape(ans.size)
        diff = - (t - ans)
        diff = diff.reshape((diff.size, 1))

        diff = np.dot(diff, one_w.T)
        diff = diff.reshape((n * i, 1))
        dw = np.dot(xx.T, diff)
        w -= lr * dw
    ans = np.dot(xx, w)
    ans = ans.reshape((nn, 4))
    ans = np.sum(ans, axis=1)
    print("pre:", ans)
