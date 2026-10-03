"""M16 规划器模块的输入输出类型定义（接口契约）。
M16 planner — input/output type definitions (interface contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


class PlannerError(ValueError):
    """非法构造参数、goal 维度不匹配或退化参数时抛出。
    Raised on bad constructor args, goal dimension mismatch, or
    degenerate CEM parameters."""


@dataclass
class PlanResult:
    """plan() 的输出。Output of plan().

    Attributes:
        actions: 最优动作序列 (H, action_dim)。
            Best action sequence (H, action_dim).
        zs: 预测潜状态序列，长度 H+1（含初始 z）。
            Predicted latent states, length H+1 (incl. initial z).
        rewards: 每步预测回报。Predicted reward per step.
        dones: 每步预测终止概率。Predicted done prob per step.
        score: 目标函数值（回报和 - 终局目标距离惩罚）。
            Objective value (reward sum - terminal goal cost).
        history: 每代 CEM 的最优得分（收敛轨迹）。
            Best score per CEM generation (convergence trace).
    """
    actions: np.ndarray
    zs: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    score: float = 0.0
    history: List[float] = field(default_factory=list)

    @property
    def first_action(self) -> np.ndarray:
        """MPC 风格：只执行第一步，其余留待重规划。
        MPC-style: execute only the first action, replan after."""
        return self.actions[0]
