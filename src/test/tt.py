import numpy as np

x = np.random.randint(1, 10, (1, 2, 2))
print(x)
x = np.concatenate((x, x), axis=2)
print(x)
