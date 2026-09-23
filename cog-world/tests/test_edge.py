"""M26 边界条件测试。"""
import pytest

from cog_body import Body
from cog_world import World, WorldError


def test_invalid_constructor():
    with pytest.raises(WorldError):
        World(size=0)
    with pytest.raises(WorldError):
        World(n_food=-1)
    with pytest.raises(WorldError):
        World(n_danger=-2)


def test_unknown_agent_id():
    w = World()
    with pytest.raises(WorldError, match="unknown agent"):
        w.observe(999)
    with pytest.raises(WorldError, match="unknown agent"):
        w.step(999, {})


def test_add_agent_body_and_position_conflict():
    w = World()
    with pytest.raises(WorldError, match="not both"):
        w.add_agent(Body(), position=(1.0, 1.0))


def test_empty_world_no_food_scent():
    w = World(size=10, n_food=0, n_danger=0, seed=0)
    aid = w.add_agent()
    obs = w.observe(aid)
    assert obs[2] == 0.0  # 无食物 -> 气味 0
    assert obs[3] == 0.0  # 无危险 -> 接近度 0


def test_agent_position_clamped_inside_world():
    w = World(size=5)
    aid = w.add_agent(position=(4.9, 4.9))
    w.step(aid, {"move": (10.0, 10.0)})
    assert w.agents[aid].position == (5.0, 5.0)


def test_world_without_food_still_runs():
    w = World(size=10, n_food=0, n_danger=0, seed=0)
    aid = w.add_agent()
    for _ in range(100):
        obs, reward = w.step(aid, {"move": (0.1, 0.0)})
    assert w.agents[aid].alive


def test_agent_body_position_outside_arena():
    """Body 的 arena_size 与世界不一致时由 Body 自身约束。"""
    w = World(size=10)
    body = Body(position=(20.0, 20.0), arena_size=10.0)  # 越界初始位置
    aid = w.add_agent(body)
    obs = w.observe(aid)
    assert obs[0] > 1.0  # 观测反映真实（越界）状态，直到下一次移动被拉回
    w.step(aid, {"move": (-1.0, 0.0)})
    assert w.agents[aid].position == (10.0, 10.0)  # x/y 均被拉回边界
