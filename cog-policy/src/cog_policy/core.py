"""M05 行动策略模块主实现。

职责：从潜状态 z 生成动作分布 + 价值估计。
实现：对角高斯策略（mean 由 MLP 给出，std 由可学习参数经软限幅给出），
价值头独立输出 V(z)。使用 M04 风格的 NumPy 微型网络，接口与 torch 对齐。
"""
from __future__ import annotations

from typing import List, Mapping, Tuple

import numpy as np

from .nn import Linear, Sequential, Tanh
from .types import DiagGaussian, PolicyError

#: log_std 的硬限幅，避免 std 塌缩到 0 或爆炸
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Policy:
    """高斯策略 + 价值头。

    Args:
        latent_dim: 输入潜向量维度（来自 M04）。
        action_dim: 动作维度。
        hidden_dim: 隐层宽度。
        seed: 随机种子。
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64,
                 seed: int = 0) -> None:
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise PolicyError("latent_dim/action_dim/hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        rng = np.random.default_rng(seed)
        self.mu_net = Sequential(
            Linear(latent_dim, hidden_dim, rng), Tanh(),
            Linear(hidden_dim, action_dim, rng), Tanh(),
        )
        self.value_net = Sequential(
            Linear(latent_dim, hidden_dim, rng), Tanh(),
            Linear(hidden_dim, 1, rng),
        )
        # 可学习 log_std（与状态无关），初始化为 0 => std=1
        self.log_std = np.zeros(action_dim)
        self._dlog_std = np.zeros(action_dim)

    # ------------------------------------------------------------------ API
    def forward(self, z: np.ndarray) -> Tuple[DiagGaussian, np.ndarray]:
        """z -> (dist, value)。

        Returns:
            dist: DiagGaussian，支持 sample/log_prob/entropy/deterministic。
            value: V(z)，形状 ()（标量）或 (B,)（批量）。
        """
        z = self._coerce(z)
        mean = self.mu_net.forward(z)
        std = np.exp(np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        dist = DiagGaussian(mean=mean, std=np.broadcast_to(std, mean.shape).copy())
        v = self.value_net.forward(z)
        value = v[..., 0]
        return dist, value

    __call__ = forward

    def act(self, z: np.ndarray, rng: np.random.Generator,
            deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """采样一步动作。

        Returns:
            (action, log_prob, value)
        """
        dist, value = self.forward(z)
        if deterministic:
            action = dist.deterministic()
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample(rng)
            log_prob = dist.log_prob(action)
        v = float(value) if np.ndim(value) == 0 else value
        return action, float(log_prob), float(v)

    # ------------------------------------------------------------------ 训练
    def backward(self, grad_mean: np.ndarray, grad_value: np.ndarray,
                 grad_log_std: np.ndarray | None = None) -> None:
        """手动回传（策略梯度 + 价值回归外部已算好对输出的梯度）。"""
        self.mu_net.backward(np.asarray(grad_mean, dtype=np.float64))
        self.value_net.backward(np.asarray(grad_value, dtype=np.float64))
        if grad_log_std is not None:
            self._dlog_std += np.asarray(grad_log_std, dtype=np.float64).ravel()

    def parameters(self) -> List[np.ndarray]:
        return [*self.mu_net.parameters(), *self.value_net.parameters(),
                self.log_std]

    def zero_grad(self) -> None:
        self.mu_net.zero_grad()
        self.value_net.zero_grad()
        self._dlog_std = np.zeros_like(self.log_std)

    def sgd_step(self, lr: float) -> None:
        self.mu_net.sgd_step(lr)
        self.value_net.sgd_step(lr)
        self.log_std -= lr * self._dlog_std

    # ------------------------------------------------------------------ 存取
    def state_dict(self) -> Mapping[str, np.ndarray]:
        return {
            "latent_dim": self.latent_dim, "action_dim": self.action_dim,
            "mu_W1": self.mu_net.layers[0].W, "mu_b1": self.mu_net.layers[0].b,
            "mu_W2": self.mu_net.layers[2].W, "mu_b2": self.mu_net.layers[2].b,
            "v_W1": self.value_net.layers[0].W, "v_b1": self.value_net.layers[0].b,
            "v_W2": self.value_net.layers[2].W, "v_b2": self.value_net.layers[2].b,
            "log_std": self.log_std,
        }

    def load_state_dict(self, sd: Mapping[str, np.ndarray]) -> None:
        if sd["latent_dim"] != self.latent_dim or sd["action_dim"] != self.action_dim:
            raise PolicyError("state_dict shape mismatch")
        for net, p in ((self.mu_net, "mu"), (self.value_net, "v")):
            net.layers[0].W = np.array(sd[f"{p}_W1"], dtype=np.float64)
            net.layers[0].b = np.array(sd[f"{p}_b1"], dtype=np.float64)
            net.layers[2].W = np.array(sd[f"{p}_W2"], dtype=np.float64)
            net.layers[2].b = np.array(sd[f"{p}_b2"], dtype=np.float64)
        self.log_std = np.array(sd["log_std"], dtype=np.float64)

    # ------------------------------------------------------------------ 内部
    def _coerce(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.shape[-1] != self.latent_dim:
            raise PolicyError(f"expected last dim {self.latent_dim}, got {z.shape[-1]}")
        if not np.all(np.isfinite(z)):
            raise PolicyError("z contains NaN/Inf")
        return z
