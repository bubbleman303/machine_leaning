from src.models.neural_net import NeuralNetWork
from src.conf import config
import numpy as np
import time

nn = NeuralNetWork(input_size=784, hidden_size=50, output_size=10, depth=0, batch_size=1000, lr=0.1,
                   activation_function_mode="lr")

train_value = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_i.npy"))/255
train_label = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_t.npy"))
x = train_value.reshape((60000, 1, 28, 28))
print("hi")
print(nn.accuracy(train_value, train_label))
start = time.time()
nn.batch_train(train_value, train_label, epochs=10)
acc = nn.accuracy(train_value, train_label)
print(acc)
print(time.time()-start)