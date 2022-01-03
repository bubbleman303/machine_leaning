import numpy as np

c_num = 2
x = np.arange(10).reshape((-1, c_num))
print(x)
sums = np.sum(x, axis=1)
print(sums)
