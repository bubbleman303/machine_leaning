from src.models.layers import ReluLayer
from src.models.neural_net import NeuralNetWork

import numpy as np

r = ReluLayer()
nn = NeuralNetWork(input_size=8, hidden_size=2, output_size=1)

w = np.random.random(3)
target = np.array([3])
print(target)
for i in range(10):
    nn = NeuralNetWork(input_size=3, hidden_size=1, output_size=1, activation_function_mode="lr", depth=1, lr=0.3)
    for j in range(200):
        nn.train(w, target)
    print(nn.loss_list[-1])
