"""M04 感知编码模块的输入输出类型定义（接口契约）。"""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import numpy as np

#: 观测：向量 / 批矩阵 / 命名的多模态字典（会被拼接后编码）
Observation = Union[np.ndarray, Sequence[float], Mapping[str, Any]]


class PerceptionError(ValueError):
    """观测维度不匹配或格式非法时抛出。"""
