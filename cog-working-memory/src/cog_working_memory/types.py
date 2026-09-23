"""M06 工作记忆模块的输入输出类型定义（接口契约）。
M06 working memory — input/output type definitions (interface contract).
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np

#: 潜向量（来自 M04）或查询向量
#: Latent vector (from M04) or query vector
Vector = Union[np.ndarray, Sequence[float]]


class WorkingMemoryError(ValueError):
    """读写维度不匹配、空记忆读取等非法操作时抛出。
    Raised on dimension mismatch, reading an empty memory, and other
    illegal operations."""
