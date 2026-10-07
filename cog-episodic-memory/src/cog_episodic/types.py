"""M07 情景记忆模块的输入输出类型定义（接口契约）。
M07 episodic memory — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np

#: 情景内容：观测/潜向量（必需）+ 任意元数据（动作、回报、内部状态…）。
#: Episode content: the observation/latent vector (required) plus
#: arbitrary metadata (action, reward, internal state, ...).
EpisodeMeta = Mapping[str, Any]


class MemoryError(ValueError):
    """容量/维度非法、空记忆查询时抛出。
    Raised on invalid capacity/dimension, or queries against an
    empty memory."""


@dataclass
class Episode:
    """一条情景记忆。One episodic record.

    Attributes:
        z: 存储时的事件向量。The event vector as stored.
        t: 写入序号（单调递增，即"经历的时间"）。Monotonic write
            index (the time of the experience).
        meta: 事件元数据（action / reward / done / 内部状态等）。
            Event metadata (action / reward / done / internal
            state, ...).
    """
    z: np.ndarray
    t: int = 0
    meta: EpisodeMeta = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """读元数据字段（带默认值）。Read a metadata field with a
        default."""
        return self.meta.get(key, default)


@dataclass
class Retrieved:
    """read() 的检索结果。Result of read().

    Attributes:
        episodes: 按相似度降序的情景列表。Episodes in
            descending similarity.
        sims: 对应的余弦相似度。Matching cosine similarities.
    """
    episodes: List[Episode] = field(default_factory=list)
    sims: List[float] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.episodes)

    @property
    def best(self) -> Optional[Episode]:
        """最相似的一条（空检索时 None）。Most similar episode (None
        when nothing was retrieved)."""
        return self.episodes[0] if self.episodes else None
