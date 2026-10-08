"""M08 语义记忆模块主实现。
M08 semantic memory — main implementation.

职责：存储符号化世界知识（概念 + 带类型加权关系），并用**扩散激活**
做关联检索：给一组种子概念，激活沿边传播并随跳数衰减，返回一张
"此刻哪些概念被唤起"的激活表——认知心理学里的经典语义网络回忆。
Responsibilities: store symbolic world knowledge (concepts + typed
weighted relations) and retrieve by **spreading activation**: seed
concepts fire, activation propagates along edges with hop decay,
and the result is an activation table of "which concepts are
currently evoked" — the classic semantic-network recall of
cognitive psychology.

与 M07 情景记忆的分工：M07 存"经历"（向量，相似检索）；本模块存
"知识"（符号，图检索）。
Division of labor with M07 episodic memory: M07 stores
*experiences* (vectors, similarity retrieval); this module stores
*knowledge* (symbols, graph retrieval).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .types import Activation, Concept, Link, SemanticError

__all__ = ["SemanticMemory"]


class SemanticMemory:
    """知识图谱语义记忆。Knowledge-graph semantic memory.

    Args:
        decay: 每跳激活保留率 (0, 1]；1 表示不衰减（慎用）。
            Per-hop activation retention in (0, 1].
        threshold: 激活低于该值的概念从结果中剔除。
            Concepts below this activation are dropped from results.
    """

    def __init__(self, decay: float = 0.5,
                 threshold: float = 0.05) -> None:
        if not 0.0 < decay <= 1.0:
            raise SemanticError("decay must be in (0, 1]")
        if threshold < 0.0:
            raise SemanticError("threshold must be >= 0")
        self.decay = float(decay)
        self.threshold = float(threshold)
        self._concepts: Dict[str, Concept] = {}
        # 邻接表：src -> [(dst, weight), ...]（出边）
        # adjacency: src -> [(dst, weight), ...] (out-edges)
        self._out: Dict[str, List[Tuple[str, float]]] = {}
        self._links: List[Link] = []

    # ------------------------------------------------------------------ API
    def add_concept(self, cid: str, **attrs: Any) -> Concept:
        """新增（或更新属性）一个概念。Add (or refresh) a concept."""
        if not isinstance(cid, str) or not cid:
            raise SemanticError("concept id must be a non-empty string")
        if cid in self._concepts:
            self._concepts[cid].attrs.update(attrs)
            return self._concepts[cid]
        c = Concept(cid=cid, attrs=dict(attrs))
        self._concepts[cid] = c
        self._out.setdefault(cid, [])
        return c

    def link(self, src: str, dst: str, relation: str = "rel",
             weight: float = 1.0, bidirectional: bool = False) -> Link:
        """连接两个概念。Link two concepts.

        Args:
            bidirectional: True 时同时加反向边（共享权重）。
                Also add the reverse edge (same weight).
        """
        if src not in self._concepts or dst not in self._concepts:
            raise SemanticError(
                f"link requires existing concepts: {src!r} -> {dst!r}")
        if src == dst:
            raise SemanticError("self-loops are not allowed")
        if not 0.0 < weight <= 1.0:
            raise SemanticError("weight must be in (0, 1]")
        lk = Link(src=src, dst=dst, relation=relation, weight=float(weight))
        self._links.append(lk)
        self._out[src].append((dst, lk.weight))
        if bidirectional:
            self._links.append(Link(src=dst, dst=src,
                                    relation=relation, weight=float(weight)))
            self._out[dst].append((src, lk.weight))
        return lk

    def get(self, cid: str) -> Concept:
        """按 id 取概念。Fetch a concept by id."""
        if cid not in self._concepts:
            raise SemanticError(f"unknown concept: {cid!r}")
        return self._concepts[cid]

    def neighbors(self, cid: str) -> List[str]:
        """直接邻居（出边指向的概念）。Direct neighbors (out-edge
        targets)."""
        if cid not in self._concepts:
            raise SemanticError(f"unknown concept: {cid!r}")
        return [dst for dst, _ in self._out[cid]]

    def spread(self, seeds: Mapping[str, float] | List[str],
               depth: int = 3) -> Activation:
        """扩散激活：从种子出发传播激活并返回激活表。
        Spreading activation: fire from seeds, propagate, return the
        activation table.

        Args:
            seeds: {cid: 初始激活} 或概念 id 列表（初始激活取 1.0）。
                {cid: initial activation} or a list of ids (each
                starting at 1.0).
            depth: 最大传播跳数。Max propagation hops.
        Returns:
            {cid: 激活}，按激活值降序、低于阈值的已剔除。
                {cid: activation}, descending, below-threshold pruned.
        """
        if depth < 0:
            raise SemanticError("depth must be >= 0")
        if isinstance(seeds, Mapping):
            acts: Dict[str, float] = {}
            for cid, a in seeds.items():
                if cid not in self._concepts:
                    raise SemanticError(f"unknown seed concept: {cid!r}")
                if not np.isfinite(a) or a <= 0:
                    raise SemanticError(
                        f"seed activation must be positive & finite: {cid}")
                acts[cid] = acts.get(cid, 0.0) + float(a)
        else:
            ids = list(seeds)
            if not ids:
                raise SemanticError("seeds must not be empty")
            acts = {}
            for cid in ids:
                if cid not in self._concepts:
                    raise SemanticError(f"unknown seed concept: {cid!r}")
                acts[cid] = acts.get(cid, 0.0) + 1.0

        fired = acts.copy()
        frontier = acts
        for _ in range(depth):
            nxt: Dict[str, float] = {}
            for src, a in frontier.items():
                for dst, w in self._out[src]:
                    gain = a * w * self.decay
                    if gain >= self.threshold:
                        nxt[dst] = nxt.get(dst, 0.0) + gain
            nxt = {c: a for c, a in nxt.items()
                   if c not in fired or a > fired[c]}
            if not nxt:
                break
            for c, a in nxt.items():             # 激活只增不减 / keep max
                fired[c] = max(fired.get(c, 0.0), a)
            frontier = nxt
        out = {c: a for c, a in fired.items() if a >= self.threshold}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, cid: str) -> bool:
        return cid in self._concepts

    # ------------------------------------------------------- save / restore
    def state_dict(self) -> Mapping[str, Any]:
        """快照（纯数据，可 pickle / 交给 M03 存档）。Snapshot (pure
        data; picklable / hand to M03)."""
        return {
            "decay": self.decay,
            "threshold": self.threshold,
            "concepts": [{"cid": c.cid, "attrs": dict(c.attrs)}
                         for c in self._concepts.values()],
            "links": [{"src": l.src, "dst": l.dst,
                       "relation": l.relation, "weight": l.weight}
                      for l in self._links],
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self._concepts.clear()
        self._out.clear()
        self._links.clear()
        self.decay = float(sd["decay"])
        self.threshold = float(sd["threshold"])
        for c in sd["concepts"]:
            self.add_concept(c["cid"], **c["attrs"])
        for l in sd["links"]:
            self.link(l["src"], l["dst"], l["relation"], l["weight"])
