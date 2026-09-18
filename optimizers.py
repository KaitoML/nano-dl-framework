import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def update(self):
        raise NotImplementedError('this method is not implemented')

    def zero_grad(self):
        raise NotImplementedError('this method is not implemented')


class SGD(Optimizer):
    def __repr__(self):
        return f'SGD(lr={self.lr})'

    def update(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)


class Adam(Optimizer):
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        # Initialize adam parameters
        self.t = 0
        self.v = [np.zeros(p.shape) for p in self.parameters]
        self.s = [np.zeros(p.shape) for p in self.parameters]

    def __repr__(self):
        return f'Adam(lr={self.lr})'

    def update(self):
        self.t += 1

        for i in range(len(self.v)):
            self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * self.parameters[i].grad

        for i in range(len(self.s)):
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * self.parameters[i].grad ** 2

        v_corr = [v / (1 - self.beta1 ** self.t) for v in self.v]
        s_corr = [s / (1 - self.beta2 ** self.t) for s in self.s]

        for i, parameter in enumerate(self.parameters):
            parameter.data -= self.lr * v_corr[i] / ((s_corr[i]) + self.eps) ** 0.5

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)