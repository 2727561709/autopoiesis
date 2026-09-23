"""M03 存档模块的输入输出类型定义（接口契约）。
M03 checkpoint module — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: 当前存档格式版本。加载时用于兼容性检查。
#: Current archive format version; used for compatibility checks on load.
FORMAT_VERSION = 1


@dataclass
class CheckpointMeta:
    """存档元数据，随状态一起持久化。
    Archive metadata, persisted alongside the state."""

    format_version: int = FORMAT_VERSION
    created_at: float = 0.0
    module_versions: Dict[str, str] = field(default_factory=dict)
    note: Optional[str] = None


class CheckpointError(RuntimeError):
    """存档读写或版本兼容性失败时抛出。
    Raised on archive read/write failure or version incompatibility."""
