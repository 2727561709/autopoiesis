"""M15 目标管理模块的输入输出类型定义（接口契约）。
M15 goal management — input/output type definitions (interface contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

#: 目标状态 / goal statuses
ACTIVE = "active"
COMPLETED = "completed"
EXPIRED = "expired"

VALID_STATUS = (ACTIVE, COMPLETED, EXPIRED)


@dataclass
class Goal:
    """一个目标。A single goal.

    Attributes:
        name: 目标名（如 "satisfy_energy"）。Goal name (e.g.
            "satisfy_energy").
        priority: 优先级 ∈ [0, 1]，越高越先被 current() 选中。
            Priority ∈ [0, 1]; higher is picked first by current().
        deadline: 过期步数（None = 永不过期）。Expiry step (None =
            never expires).
        status: active / completed / expired。
        metadata: 任意附加信息（来源、参数等）。Any extra info
            (source, parameters, ...).
    """

    name: str
    priority: float = 0.5
    deadline: Optional[int] = None
    status: str = ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


#: 驱动缺口（来自 M12 DriveSystem.deficit()）
#: Drive deficits (from M12 DriveSystem.deficit())
Deficits = Mapping[str, float]


class GoalManagerError(ValueError):
    """目标或参数非法时抛出。
    Raised on an illegal goal or parameter."""
