import numpy as np


class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for m, v, param, grad in zip(self.m, self.v, params, grads):
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad ** 2 - v)

            param -= lr_t * m / (np.sqrt(v) + 1e-7)


class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for h, param, grad in zip(self.h, params, grads):
            h += grad ** 2
            param -= self.lr * grad / (np.sqrt(h) + 1e-7)


class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad
