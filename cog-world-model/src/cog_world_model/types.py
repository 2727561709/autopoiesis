"""M10 认知世界模型模块的输入输出类型定义（接口契约）。
M10 cognitive world model — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Union

import numpy as np

#: 动作：离散动作号（int，内部 one-hot 化）/ 连续向量 / 批矩阵。
#: Action: discrete action id (int, one-hot encoded internally) /
#: continuous vector / batch matrix.
Action = Union[int, np.ndarray, Sequence[float]]


class WorldModelError(ValueError):
    """维度不匹配、非法输入或空缓冲区训练时抛出。
    Raised on dimension mismatch, malformed input, or training on an
    empty buffer."""


@dataclass
class ImagineResult:
    """imagine() 的想象 rollout 结果。Result of an imagine() rollout.

    Attributes:
        zs: 潜状态序列，长度 H+1（含初始 z）。Latent-state sequence,
            length H+1 (including the initial z).
        actions: 每步采取的动作（one-hot 后的向量）。Action taken at
            each step (one-hot vector form).
        rewards: 每步预测的即时回报。Predicted immediate reward per step.
        dones: 每步预测的终止概率。Predicted termination prob per step.
    """
    zs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        """累计预测回报。Sum of predicted rewards."""
        return float(sum(self.rewards))

    @property
    def terminated(self) -> bool:
        """rollout 是否因预测 done > 0.5 提前终止。
        Whether the rollout stopped early on predicted done > 0.5."""
        return bool(self.dones) and self.dones[-1] > 0.5
