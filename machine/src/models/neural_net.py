import json
import numpy as np
from src.models import layers
from src.conf import config
import matplotlib.pyplot as plt


class NeuralNetWork:
    def __init__(self, lr=0.01, batch_size=300, input_size=None, hidden_size=None, output_size=None, depth=0,
                 weight_init_std=0.01,
                 load_nn_name=None, activation_function_mode="r"):
        self.layers = []
        self.lr = lr
        self.batch_size = batch_size
        self.last_layer = None
        self.loss_list = []
        self.activation_mode = activation_function_mode
        if load_nn_name:
            self.load_nn(load_nn_name)
            return
        self.weight_init_std = weight_init_std
        self.layers.append(layers.AffineLayer(self.weight_init(input_size, hidden_size), np.zeros(hidden_size)))
        self.layers.append(self.call_activation())
        for i in range(depth):
            self.layers.append(layers.AffineLayer(self.weight_init(hidden_size, hidden_size), np.zeros(hidden_size)))
            self.layers.append(self.call_activation())
        self.layers.append(layers.AffineLayer(self.weight_init(hidden_size, output_size), np.zeros(output_size)))
        self.layers.append(self.call_activation())
        self.last_layer = layers.MeanSquareLoss()

    @staticmethod
    def weight_init(i, o):
        return np.random.normal(scale=1 / np.sqrt(i), size=(i, o))

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        else:
            t = np.array([np.argmax(t, axis=1)])
        return np.sum(y == t) / y.size

    def train(self, x, t):
        self.loss_list.append(self.loss(x, t))
        d_out = 1
        d_out = self.last_layer.backward(d_out)
        for layer in self.layers[::-1]:
            d_out = layer.backward(d_out)
        for layer in self.layers:
            if type(layer) == layers.AffineLayer:
                layer.w -= layer.dw * self.lr
                layer.b -= layer.db * self.lr

    def save_nn(self, name):
        param_dic = {"lr": self.lr, "batch_size": self.batch_size, "net": []}
        aff_num = 0
        last_type = ""
        if type(self.last_layer) == layers.SoftmaxWithLoss:
            last_type = "sf"
        param_dic["loss_layer"] = last_type
        for layer in self.layers:
            layer_type = type(layer)
            lt = ""
            if layer_type == layers.ReluLayer:
                lt = "relu"
            elif layer_type == layers.SigmoidLayer:
                lt = "sigmoid"
            elif layer_type == layers.AffineLayer:
                lt = "affine"
            param_dic["net"].append(lt)
            if lt == "affine":
                aff_num += 1
                np.save(config.NN_PARAM_DIR.format(f"{name}_w_{aff_num}"), layer.w)
                np.save(config.NN_PARAM_DIR.format(f"{name}_b_{aff_num}"), layer.b)
        with open(config.NN_PARAM_DIR.format(f"{name}_params.json"), "wt") as f:
            json.dump(param_dic, f)

    def load_nn(self, name):
        with open(config.NN_PARAM_DIR.format(f"{name}_params.json"), "rt") as f:
            param_dic = json.load(f)
        self.lr = param_dic["lr"]
        self.batch_size = param_dic["batch_size"]
        aff_num = 0
        for layer_str in param_dic["net"]:
            obj = None
            if layer_str == "relu":
                obj = layers.ReluLayer()
            elif layer_str == "sigmoid":
                obj = layers.SigmoidLayer()
            elif layer_str == "affine":
                aff_num += 1
                w = np.load(config.NN_PARAM_DIR.format(f"{name}_w_{aff_num}.npy"))
                b = np.load(config.NN_PARAM_DIR.format(f"{name}_b_{aff_num}.npy"))
                obj = layers.AffineLayer(w, b)
            self.layers.append(obj)
        if param_dic["loss_layer"] == "sf":
            self.last_layer = layers.SoftmaxWithLoss()

    def batch_train(self, x, t, epochs=5):
        x = self.standardization(x)
        self.loss_list = []
        for epoch in range(epochs):
            index = np.random.permutation(np.arange(x.shape[0]))
            for i in range(0, x.shape[0], self.batch_size):
                data = x[index[i:i + self.batch_size]]
                target = t[index[i:i + self.batch_size]]
                self.train(data, target)
        plt.plot(np.arange(len(self.loss_list)), self.loss_list)
        plt.show()

    def call_activation(self):
        if self.activation_mode == "r":
            return layers.ReluLayer()
        if self.activation_mode == "s":
            return layers.SigmoidLayer()
        if self.activation_mode == "lr":
            return layers.LeakyReluLayer()

    @staticmethod
    def standardization(x):
        x_mean = x.mean()
        x_std = x.std()
        out = (x - x_mean) / x_std
        return out


'''
initの段階ではlast_layerのみを作るようにする。
間の層を後から自由に追加できるようにする。（エリアを作成する）
また、層の名前も自由に保存ができるように変える。
プーリング層の実装
mnistでテスト→オセロで機械学習を実装
'''


