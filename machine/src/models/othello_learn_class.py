from machine.src.models import layers, functions as fs
from machine.src.models.one_output import OneOutputNet
from machine.src.models import optimizers
from machine.src.models.bond import Bond
import numpy as np


class OthelloLearn:
    def __init__(self, lr=0.03, activation_mode="lr", optimizer="adam", depth=2, hidden_size=16, loss_layer="ms",
                 ):

        self.lr = lr
        self.activation_mode = activation_mode
        self.optimizer_str = optimizer
        self.depth = depth
        self.hidden_size = hidden_size

        # 最初の入力を扱う層
        # 1出力のものが集められる
        self.bond = None
        self.layers = []
        self.loss_layer = None
        self.set_loss_layer(loss_layer)
        self.optimizer = self.set_optimizer(optimizer)
        self.loss_list = []

        pass

    def _get_one_output(self, i, c_num):
        """
        1つの入力を得る関数
        :param i: 入力ノード数
        :param c_num: チャンネル数（分割用に使用）
        :return: one_outputクラス
        """
        net = OneOutputNet(lr=self.lr, activation_mode=self.activation_mode, optimizer=self.optimizer_str)
        net.add_affine(i, self.hidden_size)
        net.add_activation()
        net.add_batch()
        for _ in range(self.depth):
            net.add_affine(self.hidden_size, self.hidden_size)
            net.add_activation()
            net.add_batch()
        net.add_affine(self.hidden_size, 1)
        net.add_one_affine(c_num=c_num)
        net.add_activation()
        return net

    def _add_affine(self, i, o):
        w = fs.weight_init(i, o)
        b = np.zeros(o)
        self.layers.append(layers.AffineLayer(w, b))

    def set_bond(self, o, *args: int):
        if len(args) % 2 != 0:
            raise ValueError(f"param nums must be even,got{len(args)}")
        self.bond = Bond(o, [self._get_one_output(i, k) for i, k in np.array(args).reshape(-1, 2)])

    def set_loss_layer(self, layer_type):
        if layer_type == "sf":
            self.loss_layer = layers.SoftmaxWithLoss()
        elif layer_type == "ms":
            self.loss_layer = layers.MeanSquareLoss()

    def set_layers(self, i, o, depth=2):
        self._add_affine(i, o)
        self.layers.append(layers.BatchNormalization())
        self.layers.append(layers.LeakyReluLayer())
        for i in range(depth):
            self._add_affine(o, o)
            self.layers.append(layers.BatchNormalization())
            self.layers.append(layers.LeakyReluLayer())
        self._add_affine(o, 1)

    def set_optimizer(self, opt_type):
        if opt_type == "adam":
            return optimizers.Adam(lr=self.lr)
        elif opt_type == "ada_grad":
            self.optimizer = optimizers.AdaGrad(lr=self.lr)
        elif opt_type == "sgd":
            self.optimizer = optimizers.SGD(lr=self.lr)
        else:
            raise ValueError("Choose Optimizer from (adam,ada_grad,sgd)")

    def predict(self, x, train_flag=False):
        x = self.bond.forward(x, train_flag)
        for layer in self.layers:
            if type(layer) == layers.BatchNormalization:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x

    def train(self, x, t):
        y = self.predict(x, train_flag=True)
        loss = self.loss_layer.forward(y, t)
        self.loss_list.append(loss)
        d_out = self.loss_layer.backward()
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
        self.bond.train(d_out)
