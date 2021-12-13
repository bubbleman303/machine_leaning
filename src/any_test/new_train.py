import numpy as np
import time

from src.models.convolution_neural_network import ConvolutionNeuralNetwork
from src.conf import config

train_value = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_i.npy"))
train_label = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_t.npy"))

train_value = train_value.reshape((60000, 1, 28, 28)) / 255

ccn = ConvolutionNeuralNetwork(load_nn_name="new_mnist")
start = time.time()
ccn.batch_train(train_value, train_label, epochs=10, save_param_name="new_mnist")
print(time.time() - start)
