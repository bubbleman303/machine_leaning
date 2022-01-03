import os
import json
import numpy as np
from src.models import layers
from src.conf import config
from src.models import functions as fs
from src.models import optimizers
import matplotlib.pyplot as plt


class ConvolutionNeuralNetwork:
    def __init__(self, lr=0.001, batch_size=300, load_nn_name=None, activation_function_mode="r", optimizer="adam"):
        self.layers = []
        self.lr = lr
        self.batch_size = batch_size
        self.loss_list = []
        self.activation_mode = activation_function_mode
        self.now_channel_num = None
        self.last_layer = None
        self.layer_name_dict = {
            layers.ReluLayer: "Relu",
            layers.AffineForConvLayer: "Affine",
            layers.SigmoidLayer: "Sigmoid",
            layers.Pooling: "Pooling",
            layers.LeakyReluLayer: "LeakyRelu",
            layers.Conv: "Convolution",
            layers.MeanSquareLoss: "MeanSquareLoss",
            layers.SoftmaxWithLoss: "SoftMaxWithLoss",
            layers.BatchNormalization: "BatchNormalization"
        }
        self.has_wb_list = sum([[k, self.layer_name_dict[k]] for k in (layers.Conv, layers.AffineForConvLayer)], [])
        self.has_param_list = sum(
            [[k, self.layer_name_dict[k]] for k in (layers.Conv, layers.Pooling, layers.BatchNormalization)], [])
        self.name_layer_dict = {v: k for k, v in self.layer_name_dict.items()}
        self.epoch = None
        self.now_loop = None
        self.save_param_name = None
        self.loaded = False
        if load_nn_name:
            self.load_nn(load_nn_name)
            self.loaded = True
        self.optimizer = None
        self.set_optimizer(optimizer)


    @staticmethod
    def weight_init(i, o):
        return np.random.normal(scale=1 / np.sqrt(i), size=(i, o))

    def predict(self, x, train_flag=False, show_percentage=False, title=None):
        for layer in self.layers:
            if type(layer) == layers.BatchNormalization:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        if show_percentage:
            x = np.round(fs.softmax(x)[0] * 100, 2)
            if not title:
                title = np.arange(x.size)
            score = ",".join([f"{k}:{p}%" for p, k in zip(x, title)])
            ind_list = sorted([(i, v) for i, v in enumerate(x)], key=lambda t: t[1])[::-1]
            for i in range(3):
                print(f"{ind_list[i][0]}:{ind_list[i][1]}%")
            print(score)

        return x

    def loss(self, x, t):
        y = self.predict(x, train_flag=True)
        if self.now_loop % 1000 == 0:
            print(f"epoch:{self.epoch} loop:{self.now_loop} accuracy:{self.mini_accuracy(y, t)}")
        if self.save_param_name and self.now_loop % 2000 == 0:
            self.save_nn(self.save_param_name)
        loss = self.last_layer.forward(y, t)
        return loss

    def accuracy(self, x, t, train_flag=False):
        now_sum = 0
        for i in range(0, x.shape[0], self.batch_size):
            data = x[i:i + self.batch_size]
            target = t[i:i + self.batch_size]
            y = self.predict(data, train_flag=train_flag)
            y = np.argmax(y, axis=1)
            if target.ndim != 1:
                target = np.argmax(target, axis=1)
            else:
                target = np.array([np.argmax(target, axis=1)])
            now_sum += np.sum(y == target)
        return now_sum / x.shape[0]

    @staticmethod
    def mini_accuracy(y, t):
        return np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / y.shape[0]

    def _train(self, x, t):
        loss = self.loss(x, t)
        self.loss_list.append(loss)
        d_out = 1
        d_out = self.last_layer.backward(d_out)
        for layer in self.layers[::-1]:
            d_out = layer.backward(d_out)
        params = []
        grads = []
        for layer in self.layers:
            if type(layer) in [layers.AffineLayer, layers.Conv, layers.AffineForConvLayer]:
                params.append(layer.w)
                grads.append(layer.dw)
                params.append(layer.b)
                grads.append(layer.db)
        self.optimizer.update(params, grads)

    def batch_train(self, x, t, epochs=5, save_param_name=None):
        if save_param_name is not None and not self.loaded:
            # パラメータが既に存在し、このバッチtrainが新規に作成されようとしたときにアラートを出す
            if not os.path.exists(config.NN_PARAM_DIR.format(f"{save_param_name}_w_1.npy")):
                pass
            else:
                if input(
                        f"{save_param_name} is already exist.If you run this _train,"
                        f" previous files will be removed,OK?\nEnter 'y' if continue.") != "y":
                    if input(f"Load {save_param_name} and train start? Press enter if yes") == "":
                        self.load_nn(save_param_name)
                    else:
                        print("Train canceled.")
                        return
        if self.last_layer is None:
            raise ValueError("Last layer is not set yet.")
        self.save_param_name = save_param_name
        self.loss_list = []
        for epoch in range(epochs):
            self.epoch = epoch
            index = np.random.permutation(np.arange(x.shape[0]))
            for i in range(0, x.shape[0], self.batch_size):
                self.now_loop = i
                data = x[index[i:i + self.batch_size]]
                target = t[index[i:i + self.batch_size]]
                self._train(data, target)
            test_index = np.random.permutation(np.arange(x.shape[0])[:self.batch_size])
            print(f"{'=' * 10}epoch{epoch}done{'=' * 10}")
            y = self.predict(x[test_index], train_flag=False)
            print(f"accuracy_score:{self.mini_accuracy(y, t[test_index])}")
        if save_param_name:
            self.save_nn(save_param_name)
        # plt.plot(np.arange(len(self.loss_list)), self.loss_list)
        # plt.show()

    def add_activation(self):
        if self.activation_mode == "r":
            self.layers.append(layers.ReluLayer())
        elif self.activation_mode == "s":
            self.layers.append(layers.SigmoidLayer())
        elif self.activation_mode == "lr":
            self.layers.append(layers.LeakyReluLayer())

    def add_cn(self, cn=None, filter_num=16, filter_size=3):
        if self.now_channel_num is not None:
            cn = self.now_channel_num
        else:
            self.now_channel_num = 1
        self.layers.append(layers.Conv(cn, filter_num, filter_size))
        self.now_channel_num = filter_num

    def add_pool(self, pooling_num):
        self.layers.append(layers.Pooling(pooling_num))

    def add_affine(self, x, output_size):
        last_shape = self.shape_summary(x, print_summary=False)[1:]
        input_size = 1
        for i in last_shape:
            input_size *= i
        w = fs.weight_init(input_size, output_size)
        b = np.zeros(output_size)
        self.layers.append(layers.AffineForConvLayer(w, b))

    def add_batch_normal(self):
        self.layers.append(layers.BatchNormalization())

    def set_last_layer(self, layer_type):
        if layer_type == "sf":
            self.last_layer = layers.SoftmaxWithLoss()
        elif layer_type == "ms":
            self.last_layer = layers.MeanSquareLoss()

    def set_optimizer(self, opt_type):
        if opt_type == "adam":
            self.optimizer = optimizers.Adam(lr=self.lr)
        elif opt_type == "ada_grad":
            self.optimizer = optimizers.AdaGrad(lr=self.lr)
        elif opt_type == "sgd":
            self.optimizer = optimizers.SGD(lr=self.lr)
        else:
            raise ValueError("Choose Optimizer from (adam,ada_grad,sgd)")

    def shape_summary(self, x, print_summary=True):
        if not type(x) == tuple:
            x_shape = x.shape
        else:
            x_shape = x
        for layer in self.layers:
            type_of_layer = type(layer)
            if type_of_layer == layers.Conv:
                y_h = layer.get_output_size(x_shape[2], layer.fil_h)
                y_w = layer.get_output_size(x_shape[3], layer.fil_w)
                next_shape = (x_shape[0], layer.fn, y_h, y_w)
            elif type_of_layer == layers.Pooling:
                y_h = layer.get_output_size(x_shape[2])
                y_w = layer.get_output_size(x_shape[3])
                next_shape = (x_shape[0], x_shape[1], y_h, y_w)
            elif type_of_layer == layers.AffineForConvLayer:
                next_shape = (x_shape[0], layer.w.shape[1])
            else:
                next_shape = x_shape
            if print_summary:
                print(f"{self.layer_name_dict[type_of_layer]} In:{x_shape} Out:{next_shape}")
            x_shape = next_shape
        return x_shape

    def save_nn(self, name):
        param_dic = {"lr": self.lr, "batch_size": self.batch_size, "net": [], "conv_pool_param": []}
        aff_num = 0
        batch_num = 0
        param_dic["loss_layer"] = self.layer_name_dict[type(self.last_layer)]
        for layer in self.layers:
            layer_type = type(layer)
            param_dic["net"].append(self.layer_name_dict[layer_type])
            if layer_type in self.has_wb_list:
                aff_num += 1
                np.save(config.NN_PARAM_DIR.format(f"{name}_w_{aff_num}"), layer.w)
                np.save(config.NN_PARAM_DIR.format(f"{name}_b_{aff_num}"), layer.b)
            elif layer_type == layers.BatchNormalization:
                batch_num += 1
                np.save(config.NN_PARAM_DIR.format(f"{name}_running_mean_{batch_num}"), layer.running_mean)
                np.save(config.NN_PARAM_DIR.format(f"{name}_running_var_{batch_num}"), layer.running_var)
            if layer_type in self.has_param_list:
                param_dic["conv_pool_param"].append(layer.get_params())
        with open(config.NN_PARAM_DIR.format(f"{name}_params.json"), "wt") as f:
            json.dump(param_dic, f)

    def load_nn(self, name):
        with open(config.NN_PARAM_DIR.format(f"{name}_params.json"), "rt") as f:
            param_dic = json.load(f)
        self.lr = param_dic["lr"]
        self.batch_size = param_dic["batch_size"]
        self.last_layer = self.name_layer_dict[param_dic["loss_layer"]]()
        aff_num = 0
        param_num = 0
        batch_num = 0
        for layer_str in param_dic["net"]:
            if layer_str in self.has_param_list:
                layer = self.name_layer_dict[layer_str](**param_dic["conv_pool_param"][param_num])
                param_num += 1
            elif layer_str == "BatchNormalization":
                layer = self.name_layer_dict[layer_str](**param_dic["conv_pool_param"][param_num])
                param_num += 1
            else:
                layer = self.name_layer_dict[layer_str]()
            if layer_str in self.has_wb_list:
                aff_num += 1
                w = np.load(config.NN_PARAM_DIR.format(f"{name}_w_{aff_num}.npy"))
                b = np.load(config.NN_PARAM_DIR.format(f"{name}_b_{aff_num}.npy"))
                layer.load_wb(w, b)
            elif layer_str == "BatchNormalization":
                batch_num += 1
                running_mean = np.load(config.NN_PARAM_DIR.format(f"{name}_running_mean_{batch_num}.npy"))
                running_var = np.load(config.NN_PARAM_DIR.format(f"{name}_running_var_{batch_num}.npy"))
                layer.load_mean_var(running_mean, running_var)
            self.layers.append(layer)


'''
initの段階ではlast_layerのみを作るようにする。
間の層を後から自由に追加できるようにする。（エリアを作成する）
また、層の名前も自由に保存ができるように変える。
プーリング層の実装
mnistでテスト→オセロで機械学習を実装
'''
