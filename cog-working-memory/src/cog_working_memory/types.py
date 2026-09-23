"""M06 工作记忆模块的输入输出类型定义（接口契约）。"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

#: 潜向量（来自 M04）或查询向量
Vector = Union[np.ndarray, Sequence[float]]


class WorkingMemoryError(ValueError):
    """读写维度不匹配、空记忆读取等非法操作时抛出。"""
