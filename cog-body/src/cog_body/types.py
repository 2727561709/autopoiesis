"""M25 虚拟身体模块的输入输出类型定义（接口契约）。"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

#: 动作：{"move": (dx, dy), "eat": amount, "rest": bool, ...}
Action = Mapping[str, object]
#: 效果：drive 名 -> 增量，交给 M12 驱动系统
Effects = Dict[str, float]
#: 位置
Position = Tuple[float, float]


class BodyError(ValueError):
    """动作格式或参数非法时抛出。"""
