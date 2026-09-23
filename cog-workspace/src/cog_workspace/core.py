"""M14 全局工作空间模块主实现。

职责：多源竞争性广播，决定注意焦点。

模型（全局工作空间理论的向量化近似）：
1. 各源（感知/记忆/情绪/目标）经各自的随机投影 + tanh 映射到广播空间。
2. 竞争：全局上下文 comp = tanh(Σ hᵢ)，每个源的得分 = ⟨hᵢ, comp⟩ / √d
   （与当前整体语境越一致的源越显著）。
3. 广播：broadcast = Σ softmax(scoreᵢ / T) · hᵢ，权重即注意焦点，
   可解释、可记录（供 M17 元认知 / M23 可视化使用）。
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from .types import SOURCE_NAMES, BroadcastResult, SourceVector, WorkspaceError

_EPS = 1e-12

__all__ = ["Workspace"]


class Workspace:
    """全局工作空间。

    Args:
        source_dims: {"perc": d1, "mem": d2, ...} 至少一个源，键必须属于
            ("perc", "mem", "emo", "goal")；未配置的源在 forward 中传了会报错。
        broadcast_dim: 广播空间维度。
        seed: 投影初始化种子（可复现）。
    """

    def __init__(self, source_dims: Mapping[str, int], broadcast_dim: int = 64,
                 seed: int = 0) -> None:
        if not source_dims:
            raise WorkspaceError("source_dims must contain at least one source")
        if broadcast_dim <= 0:
            raise WorkspaceError(f"broadcast_dim must be positive, got {broadcast_dim!r}")
        for k, d in source_dims.items():
            if k not in SOURCE_NAMES:
                raise WorkspaceError(
                    f"unknown source {k!r}, must be one of {list(SOURCE_NAMES)}")
            if d <= 0:
                raise WorkspaceError(f"dim for {k!r} must be positive, got {d!r}")
        self.source_dims: Dict[str, int] = dict(source_dims)
        self.broadcast_dim = int(broadcast_dim)
        rng = np.random.default_rng(seed)
        self._proj: Dict[str, np.ndarray] = {}
        self._bias: Dict[str, np.ndarray] = {}
        for k, d in self.source_dims.items():
            self._proj[k] = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, broadcast_dim))
            self._bias[k] = np.zeros(broadcast_dim)

    # ------------------------------------------------------------------ API
    def forward(self, perc: Optional[SourceVector] = None,
                mem: Optional[SourceVector] = None,
                emo: Optional[SourceVector] = None,
                goal: Optional[SourceVector] = None,
                temperature: float = 1.0) -> BroadcastResult:
        """多源竞争广播。

        Args:
            perc/mem/emo/goal: 各源向量，None 表示该源本步不参与竞争。
            temperature: softmax 温度（>0），越小竞争越尖锐。

        Returns:
            (broadcast, weights)：广播向量 (broadcast_dim,) 与
            各源注意力权重 {"perc": w, ...}（只含本步激活的源，和为 1）。
        """
        if temperature <= 0:
            raise WorkspaceError(f"temperature must be > 0, got {temperature!r}")
        inputs = {"perc": perc, "mem": mem, "emo": emo, "goal": goal}
        active = {k: v for k, v in inputs.items() if v is not None}
        if not active:
            raise WorkspaceError("no active sources: at least one of perc/mem/emo/goal")

        hidden: Dict[str, np.ndarray] = {}
        for k, v in active.items():
            if k not in self.source_dims:
                raise WorkspaceError(f"source {k!r} not configured in source_dims")
            x = np.asarray(v, dtype=np.float64)
            if x.shape[-1] != self.source_dims[k]:
                raise WorkspaceError(
                    f"source {k!r}: expected dim {self.source_dims[k]}, got {x.shape[-1]}")
            if not np.all(np.isfinite(x)):
                raise WorkspaceError(f"source {k!r} contains NaN/Inf")
            hidden[k] = np.tanh(x @ self._proj[k] + self._bias[k])

        # 竞争：全局语境 = 各源隐表示之和的 tanh
        context = np.tanh(sum(hidden.values()))
        scores = {k: float(h @ context) / np.sqrt(self.broadcast_dim)
                  for k, h in hidden.items()}
        weights = _softmax_dict(scores, temperature)
        broadcast = sum(w * hidden[k] for k, w in weights.items())
        return np.asarray(broadcast), weights

    __call__ = forward


def _softmax_dict(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    m = max(scores.values())
    exps = {k: np.exp((s - m) / temperature) for k, s in scores.items()}
    total = sum(exps.values())
    return {k: float(e / total) for k, e in exps.items()}
