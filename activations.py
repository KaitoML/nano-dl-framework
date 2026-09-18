from layers import Layer

class ReLU(Layer):
    def __call__(self, x):
        self.out = x.relu()
        return self.out

    def __repr__(self):
        return 'ReLU()'

class Tanh(Layer):
    def __call__(self, x):
        self.out = x.tanh()
        return self.out

    def __repr__(self):
        return 'Tanh()'