"""M25 虚拟身体模块主实现。
M25 virtual body — main implementation.

职责：虚拟身体的能量、完整性、位置管理。
Responsibilities: manage the virtual body's energy, integrity, position.
- 移动消耗能量（与距离成正比），位置限制在方形竞技场内。
  Moving costs energy (proportional to distance); positions are bounded
  to a square arena.
- 休息恢复少量能量（默认不恢复完整性）。
  Resting recovers a little energy (integrity does not recover by default).
- 外部伤害（世界 M26 给出）通过 action["damage"] 施加到完整性。
  External damage (from world M26) is applied to integrity via
  action["damage"].
- step(action) 返回"效果"字典，直接喂给 M12 驱动系统。
  step(action) returns an "effects" dict fed directly into the M12
  drive system.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

from .types import Action, BodyError, Effects, Position

__all__ = ["Body"]

#: 一单位距离的移动能耗 / energy cost per unit distance
MOVE_COST_PER_UNIT = 0.02
#: 每步休息恢复的能量 / energy recovered per rest step
REST_RECOVERY = 0.005


class Body:
    """虚拟身体。
    Virtual body.

    Args:
        position: 初始位置 (x, y)。Initial position (x, y).
        energy: 初始能量 ∈ [0, 1]。Initial energy ∈ [0, 1].
        integrity: 初始完整性 ∈ [0, 1]。Initial integrity ∈ [0, 1].
        arena_size: 竞技场边长，位置被限制在 [0, arena_size]。
            Arena side length; positions are clipped to [0, arena_size].
    """

    def __init__(self, position: Position = (0.0, 0.0), energy: float = 1.0,
                 integrity: float = 1.0, arena_size: float = 10.0) -> None:
        for name, v in (("energy", energy), ("integrity", integrity)):
            if not (0.0 <= v <= 1.0):
                raise BodyError(f"{name} must be in [0, 1], got {v!r}")
        if arena_size <= 0:
            raise BodyError(f"arena_size must be positive, got {arena_size!r}")
        self.position: Position = (float(position[0]), float(position[1]))
        self.energy = float(energy)
        self.integrity = float(integrity)
        self.arena_size = float(arena_size)
        self.alive = True

    # ------------------------------------------------------------------ API
    def step(self, action: Action) -> Effects:
        """执行一步动作，返回驱动效果。
        Execute one action; returns drive effects.

        支持的动作键（可组合）：
        Supported action keys (combinable):
            move:   (dx, dy) 期望位移（会做能量/边界约束）
                    (dx, dy) desired displacement (energy/bounds applied)
            eat:    float 摄入能量（由世界 M26 判定可得后传入）
                    float energy intake (granted by world M26)
            rest:   bool 休息（恢复少量能量）
                    bool rest (recovers a little energy)
            damage: float 外部伤害（扣完整性，不消耗能量）
                    float external damage (reduces integrity, no energy cost)

        Returns:
            {"energy": Δ, "integrity": Δ}，供 M12 DriveSystem.step 使用。
            {"energy": Δ, "integrity": Δ}, for M12 DriveSystem.step.
        """
        if not self.alive:
            return {"energy": 0.0, "integrity": 0.0}

        d_energy = 0.0
        d_integrity = 0.0

        if "damage" in action:
            dmg = self._as_float(action["damage"], "damage", min_val=0.0)
            d_integrity -= dmg

        if "eat" in action:
            food = self._as_float(action["eat"], "eat", min_val=0.0)
            d_energy += food

        if "rest" in action and action["rest"]:
            d_energy += REST_RECOVERY

        if "move" in action:
            mv = action["move"]
            if (not isinstance(mv, (tuple, list)) or len(mv) != 2
                    or not all(isinstance(v, (int, float)) for v in mv)):
                raise BodyError(f"move must be (dx, dy) numbers, got {mv!r}")
            dx, dy = float(mv[0]), float(mv[1])
            # 能量不足时按剩余能量等比缩短位移
            # If energy is insufficient, scale the displacement down
            # proportionally to the remaining energy.
            dist = math.hypot(float(dx), float(dy))
            if dist > 1e-12:
                afford = min(1.0, self.energy / (dist * MOVE_COST_PER_UNIT))
                dx, dy = float(dx) * afford, float(dy) * afford
                actual = math.hypot(dx, dy)
                d_energy -= actual * MOVE_COST_PER_UNIT
                nx = min(max(self.position[0] + dx, 0.0), self.arena_size)
                ny = min(max(self.position[1] + dy, 0.0), self.arena_size)
                self.position = (nx, ny)

        # 内部状态直接更新（供传感器/世界读取）
        # Internal state updated in place (read by sensors / the world).
        self.energy = min(max(self.energy + d_energy, 0.0), 1.0)
        self.integrity = min(max(self.integrity + d_integrity, 0.0), 1.0)
        if self.energy <= 0.0 and self.integrity <= 0.0:
            self.alive = False

        return {"energy": d_energy, "integrity": d_integrity}

    def state(self) -> Dict[str, Any]:
        """身体状态快照（用于观测编码 / 日志 / 可视化）。
        Body state snapshot (for observation encoding / logging /
        visualization)."""
        return {
            "position": self.position,
            "energy": self.energy,
            "integrity": self.integrity,
            "alive": self.alive,
        }

    # -------------------------------------------------------------- internals
    @staticmethod
    def _as_float(v: object, name: str, min_val: float = 0.0) -> float:
        if not isinstance(v, (int, float)):
            raise BodyError(f"{name} must be a number, got {v!r}")
        if float(v) < min_val:
            raise BodyError(f"{name} must be >= {min_val}, got {v!r}")
        return float(v)
