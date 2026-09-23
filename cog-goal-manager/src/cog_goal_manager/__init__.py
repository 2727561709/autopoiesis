"""cog-goal-manager: M15 目标管理模块公开 API。

用法:
    from cog_goal_manager import GoalManager
    gm = GoalManager()
    gm.update_from_drives({"energy": 0.4, "integrity": 0.0})
    goal = gm.current()          # Goal(name="satisfy_energy", priority=0.4)
    gm.complete(goal.name)
"""
from .core import GoalManager
from .types import ACTIVE, COMPLETED, EXPIRED, Deficits, Goal, GoalManagerError

__all__ = ["GoalManager", "Goal", "Deficits", "GoalManagerError",
           "ACTIVE", "COMPLETED", "EXPIRED"]
__version__ = "0.1.0"
