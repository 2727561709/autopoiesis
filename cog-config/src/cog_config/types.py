"""M01 配置模块的输入输出类型定义（接口契约）。
M01 configuration module — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class ConfigError(ValueError):
    """配置加载或校验失败时抛出。
    Raised on config loading or validation failure."""


@dataclass
class ValidationResult:
    """validate() 的结构化结果。
    Structured result of validate()."""

    ok: bool
    errors: List[str] = field(default_factory=list)
