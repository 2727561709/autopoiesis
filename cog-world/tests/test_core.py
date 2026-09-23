"""M26 核心功能测试：生成、交互、重置。"""
import numpy as np
import pytest

from cog_body import Body
from cog_world import OBS_DIM, World, WorldError


def test_generation_counts():
    w = World(size=10, n_food=7, n_danger=3, seed=1)
    assert len(w.food) == 7
    assert len(w.dangers) == 3
    for fx, fy in w.food:
        assert 0 <= fx <= 10 and 0 <= fy <= 10


def test_reset_regenerates_and_is_deterministic():
    w1 = World(size=10, n_food=5, seed=42)
    w2 = World(size=10, n_food=5, seed=42)
    assert w1.food == w2.food
    w1.reset(seed=99)
    w3 = World(size=10, n_food=5, seed=99)
    assert w1.food == w3.food
    assert w1.food != w2.food


def test_add_agent_returns_incrementing_ids():
    w = World()
    a1 = w.add_agent()
    a2 = w.add_agent()
    assert a2 == a1 + 1
    assert w.agents[a1].arena_size == w.size


def test_add_existing_body():
    w = World(size=5)
    body = Body(position=(1.0, 1.0), arena_size=5)
    aid = w.add_agent(body)
    assert w.agents[aid] is body


def test_observe_shape_and_range():
    w = World(size=10, n_food=5, n_danger=2, seed=3)
    aid = w.add_agent()
    obs = w.observe(aid)
    assert obs.shape == (OBS_DIM,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_step_returns_obs_and_reward():
    w = World(size=10, n_food=0, n_danger=0, seed=0)  # 空世界保证确定性
    aid = w.add_agent()
    obs, reward = w.step(aid, {"move": (0.3, 0.0)})
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    # 一步移动的奖励：-移动能耗 - 存活开销（没吃到食物时）
    assert reward == pytest.approx(-0.3 * 0.02 - 0.01)


def test_eating_food():
    w = World(size=10, n_food=0, seed=0)
    w.food = [(5.0, 5.0)]
    aid = w.add_agent(position=(4.7, 5.0))
    body = w.agents[aid]
    body.energy = 0.5  # 避免吃食物后被裁剪到 1.0
    obs, reward = w.step(aid, {"move": (0.3, 0.0)})  # 走到 (5.0, 5.0) 吃掉食物
    assert len(w.food) == 0
    assert body.energy == pytest.approx(0.5 - 0.3 * 0.02 + 0.3)
    assert reward > 0  # 吃到食物净收益为正


def test_food_not_eaten_when_far():
    w = World(size=10, n_food=0, seed=0)
    w.food = [(9.0, 9.0)]
    aid = w.add_agent(position=(0.0, 0.0))
    w.step(aid, {"move": (0.1, 0.0)})
    assert len(w.food) == 1


def test_danger_damage():
    w = World(size=10, n_food=0, n_danger=0, seed=0)
    w.dangers = [(5.0, 5.0)]
    aid = w.add_agent(position=(4.8, 5.0))
    i0 = w.agents[aid].integrity
    obs, reward = w.step(aid, {"move": (0.3, 0.0)})  # 走进危险区
    assert w.agents[aid].integrity == pytest.approx(i0 - 0.1)
    assert reward < 0


def test_movement_updates_observation():
    w = World(size=10, n_food=5, n_danger=2, seed=3)
    aid = w.add_agent(position=(0.0, 0.0))
    obs1 = w.observe(aid)
    w.step(aid, {"move": (2.0, 0.0)})
    obs2 = w.observe(aid)
    assert obs2[0] == pytest.approx(0.2)  # x/size
    assert obs1[0] == pytest.approx(0.0)
