"""M07 情景记忆模块主实现。
M07 episodic memory — main implementation.

职责：长期存储经历过的情景（事件向量 + 元数据），并支持——
Responsibilities: long-term storage of experienced episodes (event
vector + metadata), with support for
- `read(query, k)`：余弦相似度 top-k 检索（"这像哪段经历？"）
  cosine-similarity top-k retrieval ("which past experience does
  this resemble?")
- `recent(n)`：读最近 n 条（时间近因）
  the most recent n episodes (temporal recency)
- `sample(n)`：随机回放批（训练数据来源，如 M10 世界模型）
  a random replay batch (training-data source, e.g. for the M10
  world model)

与 M06 工作记忆的分工：M06 是短期 FIFO 缓冲（几十条，按查询注意
衰减），本模块是长期库（容量大、按相似度检索、不衰减）。
Division of labor with M06 working memory: M06 is a short-term FIFO
buffer (dozens of items, attention-decayed); this module is a
long-term store (large capacity, similarity-retrieved, no decay).
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

import numpy as np

from .types import Episode, EpisodeMeta, MemoryError, Retrieved

__all__ = ["EpisodicMemory"]


class EpisodicMemory:
    """长期情景记忆库。Long-term episodic store.

    Args:
        capacity: 最大条数；写满后淘汰最旧（FIFO 淘汰，但被检索的
            旧条目仍可命中——按写入序号淘汰，不做频率加权）。
            Max entries; oldest is evicted when full.
        latent_dim: 事件向量维度。Event-vector dimension.
        seed: 随机种子（影响 sample 的可复现性）。
            Random seed (makes sample() reproducible).
    """

    def __init__(self, capacity: int = 10000, latent_dim: int = 128,
                 seed: int = 0) -> None:
        if capacity <= 0:
            raise MemoryError("capacity must be positive")
        if latent_dim <= 0:
            raise MemoryError("latent_dim must be positive")
        self.capacity = int(capacity)
        self.latent_dim = int(latent_dim)
        self._store: List[Episode] = []
        self._t = 0                              # 经历时间计数 / experience clock
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API
    def write(self, z: np.ndarray, meta: Optional[EpisodeMeta] = None,
              **kw_meta: Any) -> Episode:
        """写入一条情景（元数据可用 dict 或关键字参数）。
        Write one episode (metadata via dict or keyword args).

        Returns:
            存入的 Episode（含分配的时间戳 t）。
            The stored Episode (with its assigned timestamp t).
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 1 or z.shape[0] != self.latent_dim:
            raise MemoryError(
                f"z must be ({self.latent_dim},), got shape {z.shape}")
        if not np.all(np.isfinite(z)):
            raise MemoryError("z contains NaN/Inf")
        merged: dict = dict(meta or {})
        merged.update(kw_meta)
        ep = Episode(z=z.copy(), t=self._t, meta=merged)
        self._t += 1
        self._store.append(ep)
        if len(self._store) > self.capacity:
            self._store.pop(0)                   # FIFO 淘汰最旧 / evict oldest
        return ep

    def read(self, query: np.ndarray, k: int = 5) -> Retrieved:
        """余弦相似度 top-k 检索。Cosine-similarity top-k retrieval.

        零向量查询的相似度定义为 0（与任何向量都不相似也不相反）。
        For a zero query vector similarity is defined as 0.
        """
        if k <= 0:
            raise MemoryError(f"k must be positive, got {k}")
        q = np.asarray(query, dtype=np.float64)
        if q.ndim != 1 or q.shape[0] != self.latent_dim:
            raise MemoryError(
                f"query must be ({self.latent_dim},), got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise MemoryError("query contains NaN/Inf")
        if not self._store:
            return Retrieved()                   # 空记忆 => 空结果 / empty => empty
        mat = np.stack([ep.z for ep in self._store])       # (N, D)
        qn = np.linalg.norm(q)
        norms = np.linalg.norm(mat, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = norms * qn
            sims = np.where(denom > 0.0, (mat @ q) / np.where(
                denom > 0.0, denom, 1.0), 0.0)
        k = min(k, len(self._store))
        # 稳定降序：相似度相同时先存者靠前（可预测的检索顺序）
        # Stable descending: ties keep insertion order (predictable
        # retrieval order).
        idx = np.argsort(-sims, kind="stable")[:k]
        return Retrieved(episodes=[self._store[int(i)] for i in idx],
                         sims=[float(sims[int(i)]) for i in idx])

    def recent(self, n: int = 5) -> List[Episode]:
        """最近写入的 n 条（时间近因），按新到旧排序。
        The last n written episodes (most recent first)."""
        if n <= 0:
            raise MemoryError(f"n must be positive, got {n}")
        return list(reversed(self._store[-n:]))

    def sample(self, n: int = 32) -> List[Episode]:
        """均匀随机回放 n 条（训练数据来源）。Uniform random replay of
        n episodes (training-data source)."""
        if n <= 0:
            raise MemoryError(f"n must be positive, got {n}")
        if not self._store:
            raise MemoryError("sample on empty memory — write() first")
        idx = self._rng.integers(0, len(self._store), size=min(
            n, len(self._store)))
        return [self._store[int(i)] for i in idx]

    def clear(self) -> None:
        """清空记忆（保留时间钟不回拨，保证 t 单调）。
        Wipe the memory (the clock keeps running so t stays
        monotonic)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------- save / restore
    def state_dict(self) -> Mapping[str, Any]:
        """快照（纯 NumPy 数组 + 元数据，可 pickle / 交给 M03 存档）。
        Snapshot (pure NumPy arrays + metadata; picklable / hand to
        M03)."""
        return {
            "capacity": self.capacity,
            "latent_dim": self.latent_dim,
            "t": self._t,
            "episodes": [
                {"z": ep.z, "t": ep.t, "meta": dict(ep.meta)}
                for ep in self._store
            ],
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        if sd["latent_dim"] != self.latent_dim:
            raise MemoryError(
                f"state_dict latent_dim {sd['latent_dim']} != "
                f"{self.latent_dim}")
        self.capacity = int(sd["capacity"])
        self._t = int(sd["t"])
        self._store = [
            Episode(z=np.asarray(e["z"], dtype=np.float64),
                    t=int(e["t"]), meta=dict(e["meta"]))
            for e in sd["episodes"]
        ]
        if len(self._store) > self.capacity:
            self._store = self._store[-self.capacity:]
