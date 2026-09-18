import numpy as np
from functions import cross_entropy, softmax

class Model:
    def __repr__(self):
        raise NotImplementedError('this method is not implemented')

    def __call__(self, x):
        raise NotImplementedError('this method is not implemented')

    def parameters(self):
        return []

class Sequential(Model):
    def __init__(self, *layers):
        self.layers = layers

    def __repr__(self):
        return f'Sequential({self.layers})'

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def train(self):
        for layer in self.layers:
            layer.train()
        for parameter in self.parameters():
            parameter.requires_grad = True

    def evaluate(self):
        for layer in self.layers:
            layer.evaluate()
        for parameter in self.parameters():
            parameter.requires_grad = False

