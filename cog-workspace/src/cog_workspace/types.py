"""M14 全局工作空间模块的输入输出类型定义（接口契约）。
M14 global workspace — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np

#: 四个注意源（与 forward 的四个位置参数一一对应）
#: The four attention sources (matching forward's four positional args)
SOURCE_NAMES = ("perc", "mem", "emo", "goal")

#: 源向量（感知 z / 记忆读出 / 情绪向量 / 目标嵌入）
#: Source vector (perception z / memory readout / emotion vector / goal embedding)
SourceVector = Union[np.ndarray, Sequence[float]]
#: 广播结果与各源注意力权重
#: Broadcast result and per-source attention weights
BroadcastResult = Tuple[np.ndarray, Dict[str, float]]


class WorkspaceError(ValueError):
    """无激活源、源未配置、维度不匹配等非法操作时抛出。
    Raised when no source is active, a source is not configured,
    dimensions mismatch, etc."""
