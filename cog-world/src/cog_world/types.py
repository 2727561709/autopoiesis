"""M26 世界环境模块的输入输出类型定义（接口契约）。
M26 world environment — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from cog_body import Body

#: 智能体动作（语义见 cog-body）/ agent action (semantics in cog-body)
Action = Mapping[str, object]
#: 一步交互结果 / result of one interaction step
StepResult = Tuple[np.ndarray, float]


class WorldError(ValueError, KeyError):
    """环境参数非法或未知智能体 ID 时抛出。
    Raised on an illegal environment parameter or an unknown agent ID."""
