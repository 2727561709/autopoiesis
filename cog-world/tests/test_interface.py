"""M26 接口契约测试。"""
import inspect

import numpy as np

from cog_body import Body
from cog_world import OBS_DIM, World


def test_api_surface():
    for name in ("reset", "add_agent", "observe", "step"):
        assert hasattr(World, name)


def test_signatures():
    assert list(inspect.signature(World.observe).parameters)[:2] == ["self", "agent_id"]
    params = list(inspect.signature(World.step).parameters)
    assert params[:3] == ["self", "agent_id", "action"]


def test_step_contract():
    w = World(seed=0)
    aid = w.add_agent()
    obs, reward = w.step(aid, {})
    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)


def test_multiple_agents_isolated():
    w = World(size=10, n_food=5, seed=1)
    a1 = w.add_agent(position=(0.0, 0.0))
    a2 = w.add_agent(position=(9.0, 9.0))
    w.step(a1, {"move": (1.0, 0.0)})
    assert w.agents[a2].position == (9.0, 9.0)  # a2 不受影响


def test_agent_uses_body_semantics():
    """世界里的智能体就是 M25 的 Body（模块间只靠接口）。"""
    w = World()
    aid = w.add_agent()
    assert isinstance(w.agents[aid], Body)
