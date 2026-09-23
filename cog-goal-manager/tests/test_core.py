"""M15 核心功能测试：优先级、生成、完成、过期。"""
import pytest

from cog_goal_manager import ACTIVE, COMPLETED, EXPIRED, Goal, GoalManager


def test_add_and_current():
    gm = GoalManager()
    gm.add(Goal(name="explore", priority=0.3))
    gm.add(Goal(name="survive", priority=0.9))
    assert gm.current().name == "survive"


def test_priority_ordering_tie_breaks_by_recency():
    gm = GoalManager()
    gm.add(Goal(name="first", priority=0.5))
    gm.add(Goal(name="second", priority=0.5))
    # 优先级相同时，先加入的胜出（稳定）
    assert gm.current().name == "first"


def test_update_from_drives_generates_goals():
    gm = GoalManager()
    updated = gm.update_from_drives({"energy": 0.4, "integrity": 0.05, "social": 0.8})
    names = {g.name for g in updated}
    # 缺口 0.05 低于默认阈值 0.1，不生成
    assert names == {"satisfy_energy", "satisfy_social"}
    assert gm.current().name == "satisfy_social"  # 缺口最大 => 优先级最高
    assert gm.current().priority == pytest.approx(0.8)


def test_update_from_drives_priority_never_decreases():
    gm = GoalManager()
    gm.update_from_drives({"energy": 0.8})
    gm.update_from_drives({"energy": 0.2})  # 缺口变小
    g = gm._find("satisfy_energy")
    assert g.priority == pytest.approx(0.8)  # 只升不降


def test_update_from_drives_reactivates_completed():
    gm = GoalManager()
    gm.update_from_drives({"energy": 0.5})
    gm.complete("satisfy_energy")
    gm.update_from_drives({"energy": 0.6})  # 缺口再次升高 => 重新激活
    g = gm._find("satisfy_energy")
    assert g.status == ACTIVE
    assert g.priority == pytest.approx(0.6)


def test_complete():
    gm = GoalManager()
    gm.add(Goal(name="a", priority=0.9))
    gm.add(Goal(name="b", priority=0.5))
    assert gm.complete("a") is True
    assert gm._find("a").status == COMPLETED
    assert gm.current().name == "b"  # 完成后轮到次优先级
    assert gm.complete("nonexistent") is False


def test_deadline_expiry():
    gm = GoalManager()
    gm.add(Goal(name="urgent", priority=0.9, deadline=5))
    gm.add(Goal(name="slow", priority=0.3))
    assert gm.current(step=4).name == "urgent"
    assert gm.current(step=6).name == "slow"  # urgent 过期
    assert gm._find("urgent").status == EXPIRED


def test_expire_returns_names():
    gm = GoalManager()
    gm.add(Goal(name="x", priority=0.5, deadline=3))
    gm.add(Goal(name="y", priority=0.5))
    assert gm.expire(10) == ["x"]
    assert gm.expire(10) == []  # 幂等


def test_add_string_shorthand():
    gm = GoalManager()
    g = gm.add("find_food", priority=0.7, deadline=100)
    assert g.name == "find_food" and g.priority == 0.7


def test_add_existing_name_updates():
    gm = GoalManager()
    gm.add(Goal(name="g", priority=0.2))
    gm.add(Goal(name="g", priority=0.8))
    assert len(gm.goals) == 1
    assert gm.current().priority == pytest.approx(0.8)


def test_max_goals_evicts_lowest_priority():
    gm = GoalManager(max_goals=3)
    for i, p in enumerate([0.9, 0.5, 0.7, 0.2]):  # 第 4 个（0.2）触发驱逐
        gm.add(Goal(name=f"g{i}", priority=p))
    names = {g.name for g in gm.goals}
    assert "g3" not in names  # 最低优先级者被驱逐（而不是最早的 g0）
    assert names == {"g0", "g1", "g2"}


def test_active_sorted_desc():
    gm = GoalManager()
    gm.add(Goal(name="low", priority=0.1))
    gm.add(Goal(name="high", priority=0.9))
    gm.add(Goal(name="mid", priority=0.5))
    gm.complete("mid")
    assert [g.name for g in gm.active()] == ["high", "low"]


def test_clear():
    gm = GoalManager()
    gm.add("a")
    gm.clear()
    assert gm.current() is None
