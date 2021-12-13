import numpy as np
import matplotlib.pyplot as plt
from src.models.convolution_neural_network import ConvolutionNeuralNetwork
from src.conf import config
from PIL import Image

train_value = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_i.npy"))
train_label = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_train_t.npy"))
test_value = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_test_i.npy"))
test_label = np.load(config.TRAIN_DATA_DIR.format("mnist/mnist_test_t.npy"))

train_value = train_value.reshape((60000, 1, 28, 28)) / 255
test_value = test_value.reshape((10000, 1, 28, 28)) / 255

ccn = ConvolutionNeuralNetwork(load_nn_name="new_mnist")
ccn.shape_summary(test_value)
# print(ccn.accuracy(test_value, test_label))
idx = np.random.choice(np.arange(test_value.shape[0]))
img_array = test_value[idx][0]
# target = np.argmax(test_label[idx])
# print(target)
img = Image.open("C:/Users/Keizaburo Takashiba/Desktop/temp/c.png").convert("L").resize((28, 28))
img_array = (255 - np.asarray(img).reshape((28, 28))) / 255
plt.imshow(img_array)
plt.show()
img_array = img_array.reshape(1, 1, 28, 28)
predict = ccn.predict(img_array)
print(predict)
print(np.argmax(predict))
