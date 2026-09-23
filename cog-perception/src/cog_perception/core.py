"""M04 感知编码模块主实现。

职责：把原始观测（向量 / 批量 / 多模态字典）编码成有界潜向量 z。
输出经 tanh 有界化到 (-1, 1)，保证下游模块（记忆/世界模型）数值稳定。
"""
from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np

from .nn import Linear, Sequential, Tanh
from .types import Observation, PerceptionError

__all__ = ["Encoder"]


class Encoder:
    """观测编码器。

    Args:
        obs_dim: 观测维度（多模态字典按字段顺序拼接后的总维度）。
        latent_dim: 潜向量维度 z。
        hidden_dim: 隐层宽度。
        seed: 随机种子（同种子 => 同初始化 => 可复现）。
    """

    def __init__(self, obs_dim: int, latent_dim: int = 128, hidden_dim: int = 256,
                 seed: int = 0) -> None:
        if obs_dim <= 0 or latent_dim <= 0 or hidden_dim <= 0:
            raise PerceptionError("obs_dim/latent_dim/hidden_dim must be positive")
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        rng = np.random.default_rng(seed)
        self.net = Sequential(
            Linear(obs_dim, hidden_dim, rng),
            Tanh(),
            Linear(hidden_dim, latent_dim, rng),
            Tanh(),
        )

    # ------------------------------------------------------------------ API
    def forward(self, obs: Observation) -> np.ndarray:
        """obs -> z。支持 (obs_dim,) / (B, obs_dim) / {name: vector} 三种输入。

        Returns:
            潜向量，形状 (latent_dim,) 或 (B, latent_dim)，值域 (-1, 1)。
        """
        x = self._coerce(obs)
        if x.shape[-1] != self.obs_dim:
            raise PerceptionError(
                f"expected last dim {self.obs_dim}, got {x.shape[-1]}"
            )
        return self.net.forward(x)

    __call__ = forward

    # ------------------------------------------------------------------ 训练
    def backward(self, grad_z: np.ndarray) -> np.ndarray:
        """d L / d obs。grad_z 形状须与 forward 输出一致。"""
        return self.net.backward(np.asarray(grad_z, dtype=np.float64))

    def parameters(self) -> List[np.ndarray]:
        return self.net.parameters()

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def sgd_step(self, lr: float) -> None:
        self.net.sgd_step(lr)

    # ------------------------------------------------------------------ 存取
    def state_dict(self) -> Mapping[str, Any]:
        """权重快照（可直接 pickle / 交给 M03 存档）。"""
        return {
            "obs_dim": self.obs_dim,
            "latent_dim": self.latent_dim,
            "W1": self.net.layers[0].W, "b1": self.net.layers[0].b,
            "W2": self.net.layers[2].W, "b2": self.net.layers[2].b,
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        if sd["obs_dim"] != self.obs_dim or sd["latent_dim"] != self.latent_dim:
            raise PerceptionError(
                f"state_dict shape mismatch: "
                f"({sd['obs_dim']}->{sd['latent_dim']}) vs "
                f"({self.obs_dim}->{self.latent_dim})"
            )
        self.net.layers[0].W = np.array(sd["W1"], dtype=np.float64)
        self.net.layers[0].b = np.array(sd["b1"], dtype=np.float64)
        self.net.layers[2].W = np.array(sd["W2"], dtype=np.float64)
        self.net.layers[2].b = np.array(sd["b2"], dtype=np.float64)

    # ------------------------------------------------------------------ 内部
    def _coerce(self, obs: Observation) -> np.ndarray:
        if isinstance(obs, Mapping):
            parts = [np.asarray(v, dtype=np.float64).ravel() for v in obs.values()]
            if not parts:
                raise PerceptionError("empty multimodal observation")
            x = np.concatenate(parts)
        else:
            x = np.asarray(obs, dtype=np.float64)
        if x.ndim not in (1, 2):
            raise PerceptionError(f"obs must be 1-D or 2-D, got {x.ndim}-D")
        if not np.all(np.isfinite(x)):
            raise PerceptionError("obs contains NaN/Inf")
        return x
