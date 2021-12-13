import numpy as np

from src.models.convolution_neural_network import ConvolutionNeuralNetwork
from src.conf import config

train_value = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_i.npy"))
train_label = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_t.npy"))

train_value = train_value.reshape((60000, 1, 28, 28)) / 255

ccn = ConvolutionNeuralNetwork(lr=0.03, activation_function_mode="lr", batch_size=100)
ccn.add_cn(cn=1, filter_num=16, filter_size=3)
ccn.add_pool(2)
ccn.add_activation()
ccn.add_cn(filter_num=8)
ccn.add_pool(2)
ccn.add_activation()
ccn.add_affine(train_value.shape, 10)
ccn.set_last_layer("ms")
x = ccn.shape_summary(train_value, print_summary=True)
ccn.batch_train(train_value, train_label, epochs=30)
# print("predictstarttttttttttttttttttt")
# print(train_value)
# print(ccn.predict(train_value))
# print(train_label)

# train_value = train_value[:100]
# train_label = train_label[:100]
# ccn = ConvolutionNeuralNetwork(load_nn_name="mnist")
# print(ccn.accuracy(train_value, train_label))
