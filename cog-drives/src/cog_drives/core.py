"""M12 驱动系统模块主实现。

职责：稳态驱动的维持、衰减、恢复与危机判定。
模型：每个驱动有内部值 v ∈ [0, 1] 与设定点 s（默认 1.0）。
每个时间步 v 先按 decay 衰减（模拟持续消耗），再叠加动作效果。
缺口 deficit = clamp(s - v, 0, 1)；缺口超过阈值即危机。
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np

from .types import DRIVE_NAMES, DriveError, Effects

__all__ = ["DriveSystem"]


class DriveSystem:
    """稳态驱动系统。

    Args:
        decay: 每步衰减系数（0, 1]，1 表示不衰减。
        crisis_threshold: 缺口超过该值判定为危机（[0, 1]）。
        setpoints: 各驱动的设定点（默认全部 1.0，好奇可设低一些）。
    """

    def __init__(self, decay: float = 0.995, crisis_threshold: float = 0.15,
                 setpoints: Mapping[str, float] | None = None) -> None:
        if not (0.0 < decay <= 1.0):
            raise DriveError(f"decay must be in (0, 1], got {decay!r}")
        if not (0.0 <= crisis_threshold <= 1.0):
            raise DriveError(f"crisis_threshold must be in [0, 1], got {crisis_threshold!r}")
        self.decay = float(decay)
        self.crisis_threshold = float(crisis_threshold)
        sp = dict(setpoints or {})
        for k, v in sp.items():
            if k not in DRIVE_NAMES:
                raise DriveError(f"unknown drive: {k!r}")
            if not (0.0 <= v <= 1.0):
                raise DriveError(f"setpoint for {k!r} must be in [0, 1], got {v!r}")
        self.setpoints: Dict[str, float] = {d: float(sp.get(d, 1.0)) for d in DRIVE_NAMES}
        self.values: Dict[str, float] = {d: self.setpoints[d] for d in DRIVE_NAMES}

    # ------------------------------------------------------------------ API
    def step(self, effects: Effects | None = None) -> Dict[str, float]:
        """推进一步：先衰减，再叠加效果，最后裁剪到 [0, 1]。

        Args:
            effects: {"energy": +0.3, "curiosity": -0.1, ...}，未知键抛错。
        Returns:
            更新后的驱动值字典。
        """
        eff = dict(effects or {})
        for k in eff:
            if k not in DRIVE_NAMES:
                raise DriveError(f"unknown effect key: {k!r}")
        for d in DRIVE_NAMES:
            v = self.values[d] * self.decay + float(eff.get(d, 0.0))
            self.values[d] = float(np.clip(v, 0.0, 1.0))
        return dict(self.values)

    def deficit(self) -> Dict[str, float]:
        """各驱动的缺口（设定点 - 当前值，裁剪到 [0, 1]）。"""
        return {d: float(np.clip(self.setpoints[d] - self.values[d], 0.0, 1.0))
                for d in DRIVE_NAMES}

    def deficit_vector(self) -> np.ndarray:
        """缺口向量，形状 (5,)，顺序 = DRIVE_NAMES。"""
        return np.array([self.deficit()[d] for d in DRIVE_NAMES])

    def in_crisis(self) -> List[str]:
        """处于危机状态的驱动名列表。"""
        return [d for d, gap in self.deficit().items() if gap > self.crisis_threshold]

    def to_tensor(self) -> np.ndarray:
        """驱动值向量，形状 (5,)，顺序 = DRIVE_NAMES。"""
        return np.array([self.values[d] for d in DRIVE_NAMES], dtype=np.float64)

    def state(self) -> Dict[str, float]:
        """完整状态（值 + 缺口），用于日志/可视化。"""
        d = self.deficit()
        return {f"{k}": self.values[k] for k in DRIVE_NAMES} | {
            f"{k}_deficit": d[k] for k in DRIVE_NAMES}

    def reset(self) -> None:
        """恢复到设定点。"""
        self.values = {d: self.setpoints[d] for d in DRIVE_NAMES}
