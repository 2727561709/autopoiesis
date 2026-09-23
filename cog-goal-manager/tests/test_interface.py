"""M15 接口契约测试。"""
import inspect

from cog_goal_manager import ACTIVE, Goal, GoalManager


def test_api_surface():
    for name in ("add", "update_from_drives", "current", "complete", "expire",
                 "active", "clear"):
        assert hasattr(GoalManager, name)


def test_signatures():
    assert list(inspect.signature(GoalManager.add).parameters)[:2] == ["self", "goal"]
    params = list(inspect.signature(GoalManager.update_from_drives).parameters)
    assert params[:2] == ["self", "deficits"]
    assert list(inspect.signature(GoalManager.current).parameters) == ["self", "step"]


def test_goal_dataclass_fields():
    g = Goal(name="x")
    assert g.priority == 0.5
    assert g.deadline is None
    assert g.status == ACTIVE
    assert g.metadata == {}


def test_current_returns_goal_or_none():
    gm = GoalManager()
    assert gm.current() is None
    gm.add("x")
    assert isinstance(gm.current(), Goal)
