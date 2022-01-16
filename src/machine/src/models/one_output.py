import numpy as np
from src.models import functions as fs, layers, optimizers


class OneOutputNet:
    def __init__(self, lr=0.03, activation_mode="lr", optimizer="adam"):
        self.layers = []
        self.lr = lr
        self.activation_mode = activation_mode
        self.optimizer = None
        self.set_optimizer(optimizer)
        self.batch_size = None

    def set_optimizer(self, opt_type):
        if opt_type == "adam":
            self.optimizer = optimizers.Adam(lr=self.lr)
        elif opt_type == "ada_grad":
            self.optimizer = optimizers.AdaGrad(lr=self.lr)
        elif opt_type == "sgd":
            self.optimizer = optimizers.SGD(lr=self.lr)

    def add_affine(self, i, o):
        w = fs.weight_init(i, o)
        b = np.zeros(o)
        self.layers.append(layers.AffineLayer(w, b))

    def add_activation(self):
        if self.activation_mode == "r":
            return layers.ReluLayer()
        if self.activation_mode == "s":
            return layers.SigmoidLayer()
        if self.activation_mode == "lr":
            return layers.LeakyReluLayer()

    def add_batch(self):
        self.layers.append(layers.BatchNormalization())

    def add_one_affine(self, c_num):
        self.layers.append(layers.AddLayer(c_num))

    def forward(self, x, train_flag=False):
        if x.ndim == 3:
            x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
            self.batch_size = x.shape[0]

        for layer in self.layers:
            if type(layer) == layers.BatchNormalization:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x

    def train(self, d_out):
        d_out = d_out.reshape((-1, 1))
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
