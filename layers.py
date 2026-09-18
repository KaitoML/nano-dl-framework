import numpy as np
from engine import Tensor

class Layer:
    def __init__(self):
        self.training = False

    def __repr__(self):
        raise NotImplementedError('this method is not implemented')

    def __call__(self, x):
        raise NotImplementedError('this method is not implemented')

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def evaluate(self):
        self.training = False


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.w = Tensor(data=np.random.randn(in_features, out_features) * np.sqrt(2 / in_features), requires_grad=True)

        if bias:
            self.b = Tensor(data=np.zeros(out_features), requires_grad=True)
        else:
            self.b = None

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})'

    def __call__(self, x):
        self.out = x @ self.w

        if self.b is not None:
            self.out += self.b

        return self.out

    def parameters(self):
        return [self.w] + ([] if self.b is None else [self.b])

class BatchNorm1d(Layer):
    def __init__(self, features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.features = features
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros(features, dtype=np.float32)
        self.running_var = np.ones(features, dtype=np.float32)

        self.gamma = Tensor(data=np.ones(features), requires_grad=True)
        self.beta = Tensor(data=np.zeros(features), requires_grad=True)

    def __repr__(self):
        return f'BatchNorm1d(features={self.features})'


    def __call__(self, x):

        if self.training:
            x_mean = x.mean(axis=0, keepdims=True)
            x_var = ((x - x_mean) ** 2).mean(axis=0, keepdims=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        x_norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean.data.reshape(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var.data.reshape(-1)

        self.out = self.gamma * x_norm + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Dropout(Layer):
    def __init__(self, keep_prob=0.5):
        super().__init__()
        self.keep_prob = keep_prob

    def __repr__(self):
        return f'Dropout(keep_prob={self.keep_prob})'

    def __call__(self, x):
        if self.training:
            d = np.random.rand(*x.shape)
            d = (d < self.keep_prob)
            self.out = x * d / self.keep_prob
        else:
            self.out = x

        return self.out

