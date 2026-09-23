"""M12 驱动系统模块的输入输出类型定义（接口契约）。"""
from __future__ import annotations

from typing import Dict, List, Mapping

#: 五种稳态驱动（顺序固定，to_tensor 按此顺序）
DRIVE_NAMES: List[str] = ["energy", "integrity", "curiosity", "social", "competence"]

#: 动作效果：drive 名 -> 增量（正=满足，负=消耗）
Effects = Mapping[str, float]


class DriveError(ValueError):
    """驱动参数或效果非法时抛出。"""
