"""M15 边界条件测试。"""
import pytest

from cog_goal_manager import ACTIVE, Goal, GoalManager, GoalManagerError


def test_invalid_constructor():
    with pytest.raises(GoalManagerError):
        GoalManager(max_goals=0)
    with pytest.raises(GoalManagerError):
        GoalManager(deficit_threshold=1.5)


def test_empty_goal_name():
    with pytest.raises(GoalManagerError, match="non-empty"):
        GoalManager().add(Goal(name=""))


def test_priority_out_of_range():
    with pytest.raises(GoalManagerError, match="priority"):
        GoalManager().add(Goal(name="x", priority=1.5))
    with pytest.raises(GoalManagerError, match="priority"):
        GoalManager().add(Goal(name="x", priority=-0.1))


def test_negative_deadline():
    with pytest.raises(GoalManagerError, match="deadline"):
        GoalManager().add(Goal(name="x", deadline=-1))


def test_empty_deficits_noop():
    gm = GoalManager()
    assert gm.update_from_drives({}) == []
    assert gm.current() is None


def test_deficit_values_clamped_in_priority():
    gm = GoalManager()
    gm.update_from_drives({"energy": 5.0})  # 超过 1 的缺口
    assert gm.current().priority == 1.0


def test_negative_deficit_ignored():
    gm = GoalManager()
    gm.update_from_drives({"energy": -0.5})
    assert gm.current() is None


def test_threshold_zero_generates_any_positive_deficit():
    gm = GoalManager(deficit_threshold=0.0)
    gm.update_from_drives({"energy": 0.001})
    assert gm.current().name == "satisfy_energy"


def test_max_goals_one():
    gm = GoalManager(max_goals=1)
    gm.add(Goal(name="a", priority=0.5))
    gm.add(Goal(name="b", priority=0.9))  # 驱逐 a
    assert [g.name for g in gm.goals] == ["b"]
    assert gm.current().name == "b"


def test_expired_goal_not_current_even_if_high_priority():
    gm = GoalManager()
    gm.add(Goal(name="old", priority=1.0, deadline=0))
    gm.add(Goal(name="new", priority=0.1))
    assert gm.current(step=1).name == "new"


def test_completed_goals_not_evicting_active():
    gm = GoalManager(max_goals=2)
    gm.add(Goal(name="done", priority=0.9))
    gm.complete("done")
    gm.add(Goal(name="a", priority=0.5))
    gm.add(Goal(name="b", priority=0.6))
    # done 已完成，不占活跃名额
    assert {g.name for g in gm.active()} == {"a", "b"}
