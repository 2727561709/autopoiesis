"""M15 目标管理模块主实现。

职责：维护目标堆，从驱动缺口生成目标。
- add(goal)：同名目标已存在时更新（优先级/期限），否则入堆；
  堆满时驱逐优先级最低的活跃目标。
- update_from_drives(deficits)：对每个缺口超阈值的驱动，确保存在目标
  "satisfy_{drive}"，并把其优先级提升到缺口值（缺口越大越紧迫）。
- current(step)：先把过期目标标记 expired，再返回优先级最高的活跃目标。
- complete(name)：标记完成。
"""
from __future__ import annotations

from typing import List, Optional

from .types import ACTIVE, COMPLETED, EXPIRED, Deficits, Goal, GoalManagerError

#: 缺口超过该阈值才生成目标
DEFICIT_THRESHOLD = 0.1

__all__ = ["GoalManager"]


class GoalManager:
    """目标堆管理器。

    Args:
        max_goals: 活跃目标数上限（超出驱逐优先级最低者）。
        deficit_threshold: 驱动缺口生成目标的阈值。
    """

    def __init__(self, max_goals: int = 10,
                 deficit_threshold: float = DEFICIT_THRESHOLD) -> None:
        if max_goals <= 0:
            raise GoalManagerError(f"max_goals must be positive, got {max_goals!r}")
        if not (0.0 <= deficit_threshold <= 1.0):
            raise GoalManagerError(
                f"deficit_threshold must be in [0, 1], got {deficit_threshold!r}")
        self.max_goals = int(max_goals)
        self.deficit_threshold = float(deficit_threshold)
        self.goals: List[Goal] = []

    # ------------------------------------------------------------------ API
    def add(self, goal: Goal | str, priority: float = 0.5,
            deadline: Optional[int] = None) -> Goal:
        """添加目标；同名则更新优先级与期限并重新激活。

        Args:
            goal: Goal 对象或目标名字符串。
            priority: 优先级 [0, 1]（字符串形式时使用）。
            deadline: 过期步数（字符串形式时使用）。
        Returns:
            入堆/更新后的 Goal。
        """
        if isinstance(goal, str):
            goal = Goal(name=goal, priority=priority, deadline=deadline)
        if not goal.name or not isinstance(goal.name, str):
            raise GoalManagerError(f"goal name must be a non-empty string, got {goal.name!r}")
        if not (0.0 <= goal.priority <= 1.0):
            raise GoalManagerError(f"priority must be in [0, 1], got {goal.priority!r}")
        if goal.deadline is not None and goal.deadline < 0:
            raise GoalManagerError(f"deadline must be >= 0, got {goal.deadline!r}")

        existing = self._find(goal.name)
        if existing is not None:
            existing.priority = goal.priority
            existing.deadline = goal.deadline
            existing.status = ACTIVE
            existing.metadata = goal.metadata
            return existing

        self.goals.append(goal)
        self._evict_if_full()
        return goal

    def update_from_drives(self, deficits: Deficits) -> List[Goal]:
        """从驱动缺口生成/提升目标。

        缺口 > deficit_threshold 的驱动 d 生成目标 "satisfy_{d}"，
        优先级 = 缺口值（已存在时取 max(旧, 新) 保证紧迫性只升不降）。

        Returns:
            本步生成或更新的目标列表。
        """
        updated: List[Goal] = []
        for drive, deficit in deficits.items():
            deficit = float(deficit)
            if deficit <= self.deficit_threshold:
                continue
            name = f"satisfy_{drive}"
            priority = min(deficit, 1.0)
            existing = self._find(name)
            if existing is None:
                g = Goal(name=name, priority=priority,
                         metadata={"source": "drives", "drive": drive})
                self.add(g)
                updated.append(g)
            elif priority > existing.priority or existing.status != ACTIVE:
                existing.priority = max(existing.priority, priority)
                existing.status = ACTIVE
                updated.append(existing)
        return updated

    def current(self, step: Optional[int] = None) -> Optional[Goal]:
        """返回当前目标：先处理过期，再取优先级最高的活跃目标。

        Args:
            step: 当前步数；非 None 时 deadline < step 的目标标记 expired。
        Returns:
            Goal 或 None（无活跃目标）。
        """
        if step is not None:
            self.expire(step)
        active = [g for g in self.goals if g.status == ACTIVE]
        if not active:
            return None
        return max(active, key=lambda g: (g.priority, -self.goals.index(g)))

    def complete(self, name: str) -> bool:
        """标记目标完成。返回是否找到该目标。"""
        g = self._find(name)
        if g is None:
            return False
        g.status = COMPLETED
        return True

    def expire(self, step: int) -> List[str]:
        """把 deadline < step 的活跃目标标记过期，返回过期目标名。"""
        expired = [g.name for g in self.goals
                   if g.status == ACTIVE and g.deadline is not None and g.deadline < step]
        for g in self.goals:
            if g.name in expired:
                g.status = EXPIRED
        return expired

    def active(self) -> List[Goal]:
        """全部活跃目标（按优先级降序）。"""
        return sorted((g for g in self.goals if g.status == ACTIVE),
                      key=lambda g: g.priority, reverse=True)

    def clear(self) -> None:
        """清空所有目标。"""
        self.goals = []

    # ------------------------------------------------------------------ 内部
    def _find(self, name: str) -> Optional[Goal]:
        for g in self.goals:
            if g.name == name:
                return g
        return None

    def _evict_if_full(self) -> None:
        active = [g for g in self.goals if g.status == ACTIVE]
        while len(active) > self.max_goals:
            victim = min(active, key=lambda g: g.priority)
            self.goals.remove(victim)
            active.remove(victim)
