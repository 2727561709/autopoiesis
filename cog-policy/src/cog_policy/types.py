"""M05 行动策略模块的输入输出类型定义（接口契约）。
M05 action policy — input/output type definitions (interface contract).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class PolicyError(ValueError):
    """输入维度不匹配或分布参数非法时抛出。
    Raised on input dimension mismatch or illegal distribution
    parameters."""


@dataclass
class DiagGaussian:
    """对角高斯分布：sample / log_prob / entropy / deterministic。
    Diagonal Gaussian distribution: sample / log_prob / entropy /
    deterministic.

    与 torch.distributions.Normal 的常用子集对齐。
    Aligned with the commonly used subset of
    torch.distributions.Normal.
    """

    mean: np.ndarray   # (..., A) 或 (A,) / (..., A) or (A,)
    std: np.ndarray    # 同形状，恒正 / same shape, strictly positive

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.mean + self.std * rng.standard_normal(self.mean.shape)

    def log_prob(self, action: np.ndarray) -> np.ndarray:
        """对角高斯的对数概率（多维时为各维之和）。
        Diagonal-Gaussian log probability (summed over dims when
        multi-dimensional)."""
        a = np.asarray(action, dtype=np.float64)
        var = self.std ** 2
        lp = -0.5 * ((a - self.mean) ** 2 / var + np.log(2 * np.pi * var))
        return lp.sum(axis=-1)

    def entropy(self) -> np.ndarray:
        ent = 0.5 * np.log(2 * np.pi * np.e * self.std ** 2)
        return ent.sum(axis=-1)

    def deterministic(self) -> np.ndarray:
        return self.mean.copy()
