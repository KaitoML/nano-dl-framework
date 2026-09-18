# Nano DL Framework

A minimal deep learning framework built from scratch in pure NumPy, implementing a reverse-mode automatic differentiation (autograd) engine and a small set of composable neural network layers — no PyTorch, no TensorFlow.

This project extends the scalar-valued autograd engine from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) into a NumPy-backed `Tensor` class with full broadcasting support. Everything beyond the engine mechanism — the layers, optimizers, loss functions, and overall architecture — was designed and implemented independently.

## Features

- **`Tensor` autograd engine** (`engine.py`) — graph-based reverse-mode autodiff with broadcasting-aware gradients. Supports elementwise arithmetic (`+`, `-`, `*`, `/`), matrix multiplication, `exp`, `log`, `tanh`, `relu`, and shape-reducing ops (`sum`, `mean`, `max`), all traversed via topological-sort `backward()`.
- **Layers** (`layers.py`) — `Linear`, `BatchNorm1d` (with running statistics for train/eval modes), `Dropout`.
- **Activations** (`activations.py`) — `ReLU`, `Tanh`.
- **Model container** (`models.py`) — `Sequential`, for composing layers into a model.
- **Optimizers** (`optimizers.py`) — `SGD`, `Adam` (with bias-corrected moment estimates).
- **Loss functions** (`functions.py`) — numerically-stable `cross_entropy` (log-sum-exp trick), `softmax`.

## Validation

The framework was validated by:
- Numerically gradient-checking every differentiable operation against finite-difference approximations.
- Training a small classifier to convergence with a proper train/test split (train loss → ~0.003, test loss → ~0.008 over 1000 epochs).
- A dedicated test suite exercising each layer and activation individually.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.x. To run `tests.ipynb`, you'll also need Jupyter (or an IDE with notebook support, e.g. PyCharm, VS Code).

## Usage

```python
from engine import Tensor
from layers import Linear, BatchNorm1d, Dropout
from activations import ReLU
from models import Sequential
from optimizers import Adam
from functions import cross_entropy

model = Sequential(
    Linear(3, 16), BatchNorm1d(16), ReLU(),
    Dropout(keep_prob=0.9),
    Linear(16, 2)
)

optimizer = Adam(parameters=model.parameters(), lr=0.001)

model.train()
y_pred = model(x_train)
loss = cross_entropy(y_pred, y_train)

optimizer.zero_grad()
loss.backward()
optimizer.update()
```

See `tests.ipynb` for a full walkthrough, including a complete training loop example.

## Motivation

Built as a learning exercise to understand what happens "under the hood" of frameworks like PyTorch — implementing broadcasting, autograd, and standard layers/optimizers from first principles.

## Notes

A handful of NumPy-specific broadcasting/indexing operations (`__getitem__`, `mean`, `sum`, `max`, and the log-sum-exp `cross_entropy`) were built with AI assistance; the autograd core, layer architecture, both optimizers, and bug fixes were implemented independently.
