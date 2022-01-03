import numpy as np
from machine.src.models import functions as fs
from machine.src.models import optimizers


class Bond:
    def __init__(self, o, net_list, lr=0.03, optimizer="adam"):
        self.one_output_neural_net = net_list
        self.w = fs.weight_init(len(self.one_output_neural_net), o)
        self.b = np.zeros(o)
        self.x = None
        self.dw = None
        self.db = None
        self.lr = lr
        self.optimizer = self.set_optimizer(optimizer)

    def set_optimizer(self, opt_type):
        if opt_type == "adam":
            return optimizers.Adam(lr=self.lr)
        elif opt_type == "ada_grad":
            self.optimizer = optimizers.AdaGrad(lr=self.lr)
        elif opt_type == "sgd":
            self.optimizer = optimizers.SGD(lr=self.lr)
        else:
            raise ValueError("Choose Optimizer from (adam,ada_grad,sgd)")

    def forward(self, x, train_flag):
        self.x = np.array([net.forward(xx, train_flag=train_flag) for xx, net in zip(x, self.one_output_neural_net)]).T

        return np.dot(self.x, self.w) + self.b

    def train(self, d_out):
        dx = np.dot(d_out, self.w.T).T
        self.dw = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0)
        self.optimizer.update([self.w, self.b], [self.dw, self.db])
        for d_xx, net in zip(dx, self.one_output_neural_net):
            net.train(d_xx)
