"""内稳态奖励（移植自 M12 cog-drives 的核心思想）。

模型的唯一外部回报来源：缺口缩减。
没有任务奖励、没有人工偏好、没有"正确答案"。

模型（设定点思想来自 DriveSystem）：
    每个稳态变量 v ∈ [0, 1]，设定点 sp；
    缺口 deficit = clamp(sp - v, 0, 1)；
    奖励 = D(before) - D(after)，即一步内总缺口的缩减量。
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_SETPOINTS",
    "deficits",
    "total_deficit",
    "homeostatic_reward",
    "in_crisis",
]

# (能量设定点, 完整性设定点)
DEFAULT_SETPOINTS: Tuple[float, float] = (0.8, 0.9)


def deficits(intero: Sequence[float],
             setpoints: Sequence[float] = DEFAULT_SETPOINTS) -> np.ndarray:
    """各稳态变量的缺口向量，形状 (2,)。"""
    v = np.clip(np.asarray(intero, dtype=np.float64), 0.0, 1.0)
    sp = np.asarray(setpoints, dtype=np.float64)
    return np.clip(sp - v, 0.0, 1.0)


def total_deficit(intero: Sequence[float],
                  setpoints: Sequence[float] = DEFAULT_SETPOINTS) -> float:
    """总缺口（标量）。"""
    return float(deficits(intero, setpoints).sum())


def homeostatic_reward(before: Sequence[float],
                       after: Sequence[float],
                       setpoints: Sequence[float] = DEFAULT_SETPOINTS) -> float:
    """一步内稳态奖励：缺口缩减量。进食且存在缺口时为正，恶化时为负。"""
    return total_deficit(before, setpoints) - total_deficit(after, setpoints)


def in_crisis(intero: Sequence[float],
              threshold: float = 0.15,
              setpoints: Sequence[float] = DEFAULT_SETPOINTS) -> bool:
    """任一变量缺口超过危机阈值（对应 M12 的 in_crisis）。"""
    return bool((deficits(intero, setpoints) > threshold).any())
