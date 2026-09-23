"""M01 配置模块的输入输出类型定义（接口契约）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class ConfigError(ValueError):
    """配置加载或校验失败时抛出。"""


@dataclass
class ValidationResult:
    """validate() 的结构化结果。"""

    ok: bool
    errors: List[str] = field(default_factory=list)
