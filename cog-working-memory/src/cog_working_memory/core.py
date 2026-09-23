"""M06 工作记忆模块主实现。

职责：容量有限的短期缓存，可写可读。
模型：滑动窗口槽位（FIFO 驱逐最旧条目）+ 余弦相似度注意力读取。
read(query) 返回按注意力加权的聚合向量；attention(query) 暴露权重
（可解释、可供 M14 工作空间竞争使用）。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import Vector, WorkingMemoryError

#: 注意力温度（越小越尖锐）
TEMPERATURE = 1.0
#: 余弦相似度的数值保护
_EPS = 1e-12

__all__ = ["WorkingMemory"]


class WorkingMemory:
    """容量有限的工作记忆。

    Args:
        capacity: 槽位数量（写满后驱逐最旧条目）。
        dim: 潜向量维度；None 表示以首次写入的维度为准。
    """

    def __init__(self, capacity: int = 16, dim: Optional[int] = None) -> None:
        if capacity <= 0:
            raise WorkingMemoryError(f"capacity must be positive, got {capacity!r}")
        if dim is not None and dim <= 0:
            raise WorkingMemoryError(f"dim must be positive, got {dim!r}")
        self.capacity = int(capacity)
        self.dim = dim
        self._slots: List[np.ndarray] = []

    # ------------------------------------------------------------------ API
    def write(self, z: Vector) -> None:
        """写入一条潜向量；容量已满时驱逐最旧条目（FIFO）。"""
        v = self._coerce(z)
        if self.dim is None:
            self.dim = int(v.shape[0])
        self._slots.append(v)
        if len(self._slots) > self.capacity:
            self._slots.pop(0)

    def read(self, query: Vector) -> np.ndarray:
        """按注意力加权聚合读出。query 与槽位越相似权重越大。

        Returns:
            聚合向量，形状 (dim,)。
        """
        weights = self.attention(query)
        agg = np.zeros(self.dim)
        for w, s in zip(weights, self._slots):
            agg += w * s
        return agg

    def attention(self, query: Vector) -> np.ndarray:
        """读出注意力权重（长度 = 当前条目数，和为 1）。"""
        if not self._slots:
            raise WorkingMemoryError("read from empty working memory")
        q = self._coerce(query)
        sims = np.array([self._cosine(q, s) for s in self._slots])
        return _softmax(sims / TEMPERATURE)

    def clear(self) -> None:
        """清空全部条目（dim 保留）。"""
        self._slots = []

    def __len__(self) -> int:
        return len(self._slots)

    def state(self) -> dict:
        """状态快照（日志 / 可视化用）。"""
        return {"size": len(self._slots), "capacity": self.capacity, "dim": self.dim}

    # ------------------------------------------------------------------ 内部
    def _coerce(self, z: Vector) -> np.ndarray:
        v = np.asarray(z, dtype=np.float64)
        if v.ndim != 1:
            raise WorkingMemoryError(f"expected 1-D vector, got {v.ndim}-D")
        if self.dim is not None and v.shape[0] != self.dim:
            raise WorkingMemoryError(f"expected dim {self.dim}, got {v.shape[0]}")
        if not np.all(np.isfinite(v)):
            raise WorkingMemoryError("vector contains NaN/Inf")
        return v

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < _EPS or nb < _EPS:
            return 0.0  # 零向量与任何向量视为不相似
        return float(a @ b / (na * nb))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()  # 数值稳定
    e = np.exp(x)
    return e / e.sum()
