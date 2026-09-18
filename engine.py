import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None

    def __repr__(self):
        return f'Tensor(data={self.data})' if not self.requires_grad else f'Tensor(data={self.data}, requires_grad=True)'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx): # built with AI assistance
        out = Tensor(data=self.data[idx], requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self.grad += grad

        out._backward = _backward
        return out

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _unbroadcast(grad, target_shape):
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)

        for dim, (grad_size, target_size) in enumerate(zip(grad.shape, target_shape)):
            if target_size == 1 and grad_size != 1:
                grad = grad.sum(axis=dim, keepdims=True)

        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(data=(self.data + other.data), requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(data=(self.data * other.data), requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other))

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad += self._unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad += self._unbroadcast(grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        out = Tensor(data=(self.data * -1), requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        neg = other.__neg__()
        out = self.__add__(neg)
        return out

    def __rsub__(self, other):
        return -self + other

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise ValueError('only supported for int/float powers')

        out = Tensor(data=(self.data ** power), requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                # y = x ^ k
                # dx = k * x ^ (k-1)
                grad = power * self.data ** (power - 1) * out.grad
                self.grad += self._unbroadcast(grad, self.data.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        out = Tensor(data=(np.exp(self.data)), requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                grad = out.data * out.grad
                self.grad += self._unbroadcast(grad, self.data.shape)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(data=np.log(self.data), requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                grad = 1 / self.data * out.grad
                self.grad += self._unbroadcast(grad, self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(data=(self.data @ other.data), requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other))

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad += self._unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad += self._unbroadcast(grad, other.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False): # built with AI assistance
        out = Tensor(data=self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                     _children=(self,))

        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)

                self.grad += (np.broadcast_to(grad, self.data.shape) / n).astype(np.float32)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False): # built with AI assistance
        out = Tensor(data=self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                     _children=(self, ))

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)

                self.grad += np.broadcast_to(grad, self.data.shape).astype(np.float32)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False): # built with AI assistance
        out = Tensor(data=self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                data_max = self.data.max(axis=axis, keepdims=keepdims)
                mask = (self.data == data_max).astype(np.float32)
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)

                self.grad += mask * np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def view(self, *args):
        out = Tensor(data=self.data.reshape(*args),requires_grad=self.requires_grad, _children=(self, ))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(data=(np.maximum(self.data, 0)), requires_grad=self.requires_grad,
                     _children=(self, ))

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0) * out.grad
                self.grad += self._unbroadcast(grad, self.data.shape)

        out._backward = _backward
        return out

    def tanh(self):
        # tanh = (e^2x - 1) / (e^2x + 1)
        numerator = np.exp(2 * self.data) - 1
        denominator = np.exp(2 * self.data) + 1
        out = Tensor(data=(numerator / denominator), requires_grad=(self.requires_grad),
                     _children=(self, ))

        def _backward():
            if self.requires_grad:
                # dTanh = 1 - tanh ** 2
                grad = (1 - out.data ** 2) * out.grad
                self.grad += self._unbroadcast(grad, self.data.shape)

        out._backward = _backward
        return out

    def backward(self):
        self.grad = np.ones_like(self.data)

        topo_sort = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo_sort.append(v)

        build_topo(self)

        for component in reversed(topo_sort):
            component._backward()