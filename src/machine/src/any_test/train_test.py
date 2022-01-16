import numpy as np
import matplotlib.pyplot as plt
from src.models.one_output import OneOutputNet
from src.models.layers import MeanSquareLoss

on = OneOutputNet(lr=0.01)
n = 20
middle = 30
i = 50
x = np.random.random((n, middle, i))
ms = MeanSquareLoss()
t = np.random.randint(1, 100, n)
on.add_affine(i, 30)
on.add_activation()
on.add_affine(30, 32)
on.add_affine(32, 1)
on.add_activation()
on.add_one_affine(middle)

loss_list = []
for i in range(1000):
    out = on.forward(x)
    loss = ms.forward(out, t)
    loss_list.append(loss)
    d_out = ms.backward().reshape((n, 1))
    on.train(d_out)
xx = np.arange(len(loss_list))

plt.plot(xx, loss_list)
plt.show()
print(t)
print(on.forward(x))
print(np.min(loss_list))