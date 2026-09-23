"""微型 NumPy 神经网络（仅本模块内部使用）。
Micro NumPy neural network (internal to this module only).

提供 Linear / Tanh / Sequential：前向传播、手写反向传播、SGD 更新。
Provides Linear / Tanh / Sequential: forward pass, hand-written
backprop, SGD update.
接口命名与 torch 对齐（forward/parameters/zero_grad/backward），
后续切换 torch 后端时无需改动上层代码。
The API mirrors torch naming (forward/parameters/zero_grad/backward),
so upper layers need no changes when switching to a torch backend.
"""
from __future__ import annotations

from typing import List

import numpy as np


class Layer:
    """层基类：子类实现 forward/backward。
    Layer base class; subclasses implement forward/backward."""

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
    """全连接层 y = x @ W + b，支持批量 (B, in) 与单个 (in,) 输入。
    Fully-connected layer y = x @ W + b; supports batch (B, in) and
    single (in,) inputs."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator,
                 scale: float = 1.0) -> None:
        # Xavier 风格初始化 / Xavier-style initialization
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
        if x.ndim == 2:  # 批量 / batched
            self.dW += x.T @ grad_out
            self.db += grad_out.sum(axis=0)
        else:            # 单个 / single
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
    """tanh 激活。tanh activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = np.tanh(x)
        return self._y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_y"):
            raise RuntimeError("backward called before forward")
        return grad_out * (1.0 - self._y ** 2)


class Sequential(Layer):
    """按序组合的层序列。An ordered sequence of layers."""

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
        """所有层执行一次 SGD 更新。One SGD update for every layer."""
        for layer in self.layers:
            for p, g in zip(layer.parameters(), layer.gradients()):
                p -= lr * g
