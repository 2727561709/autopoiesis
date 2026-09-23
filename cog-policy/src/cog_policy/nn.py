"""微型 NumPy 神经网络（仅本模块内部使用）。

与 cog-perception/nn.py 保持同样的接口（forward/backward/parameters/zero_grad/sgd_step），
后续统一切换 torch 后端时一并替换。
"""
from __future__ import annotations

from typing import List

import numpy as np


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> List[np.ndarray]:
        return []

    def gradients(self) -> List[np.ndarray]:
        return []

    def zero_grad(self) -> None:
        pass


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator,
                 scale: float = 1.0) -> None:
        self.W = rng.normal(0.0, scale / np.sqrt(in_dim), size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._x
        if x is None:
            raise RuntimeError("backward called before forward")
        if x.ndim == 2:
            self.dW += x.T @ grad_out
            self.db += grad_out.sum(axis=0)
        else:
            self.dW += np.outer(x, grad_out)
            self.db += grad_out
        return grad_out @ self.W.T

    def parameters(self) -> List[np.ndarray]:
        return [self.W, self.b]

    def gradients(self) -> List[np.ndarray]:
        return [self.dW, self.db]

    def zero_grad(self) -> None:
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_y"):
            raise RuntimeError("backward called before forward")
        return grad_out * (1.0 - self._y ** 2)


class Sequential(Layer):
    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def parameters(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out

    def gradients(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.gradients())
        return out

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def sgd_step(self, lr: float) -> None:
        for layer in self.layers:
            for p, g in zip(layer.parameters(), layer.gradients()):
                p -= lr * g
