import numpy as np
import matplotlib.pyplot as plt
from src.models.othello_learn_class import OthelloLearn

ol = OthelloLearn(lr=0.5, depth=5)

ol.set_bond(16, 16, 20, 20)
ol.set_layers(16, 50, depth=5)

n = 100
x = [np.random.random((n, 16)), np.random.random((n, 20)), np.random.random((n, 20))]
target = np.random.randint(1, 64, (n, 1))
for i in range(100):
    ol.train(x, target)
print(np.mean(abs(ol.predict(x) - target)))

plt.plot(np.arange(len(ol.loss_list)), ol.loss_list)
plt.show()
print(min(ol.loss_list))
print(np.mean(ol.loss_list))
print(np.max(ol.loss_list))
