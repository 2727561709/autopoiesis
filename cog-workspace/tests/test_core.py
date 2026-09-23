"""M14 核心功能测试：竞争、广播、权重可解释。"""
import numpy as np
import pytest

from cog_workspace import Workspace, WorkspaceError

DIMS = {"perc": 16, "mem": 16, "emo": 4, "goal": 8}


def test_broadcast_shape_and_weights_sum():
    ws = Workspace(DIMS, broadcast_dim=32, seed=0)
    rng = np.random.default_rng(1)
    b, w = ws(rng.normal(size=16), rng.normal(size=16),
              rng.normal(size=4), rng.normal(size=8))
    assert b.shape == (32,)
    assert np.all(np.abs(b) <= 1.0 + 1e-9)  # tanh 隐表示的凸组合
    assert sum(w.values()) == pytest.approx(1.0)
    assert set(w) == set(DIMS)


def test_single_source_gets_full_attention():
    ws = Workspace(DIMS, broadcast_dim=32, seed=0)
    z = np.random.default_rng(2).normal(size=16)
    b, w = ws(perc=z)
    assert w == {"perc": 1.0}
    # 广播 == 该源的投影（无竞争）
    expected = np.tanh(z @ ws._proj["perc"] + ws._bias["perc"])
    assert np.allclose(b, expected)


def test_none_sources_excluded_from_weights():
    ws = Workspace(DIMS, broadcast_dim=32, seed=0)
    rng = np.random.default_rng(3)
    _, w = ws(perc=rng.normal(size=16), goal=rng.normal(size=8))
    assert set(w) == {"perc", "goal"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_weights_deterministic_same_seed():
    ws1 = Workspace(DIMS, broadcast_dim=32, seed=42)
    ws2 = Workspace(DIMS, broadcast_dim=32, seed=42)
    inputs = (np.ones(16), np.ones(16), np.ones(4), np.ones(8))
    b1, w1 = ws1(*inputs)
    b2, w2 = ws2(*inputs)
    assert np.allclose(b1, b2) and w1 == w2


def test_different_seed_different_broadcast():
    ws1 = Workspace(DIMS, broadcast_dim=32, seed=1)
    ws2 = Workspace(DIMS, broadcast_dim=32, seed=2)
    inputs = (np.ones(16), np.ones(16), np.ones(4), np.ones(8))
    b1, _ = ws1(*inputs)
    b2, _ = ws2(*inputs)
    assert not np.allclose(b1, b2)


def test_salient_source_wins_competition():
    """幅度大的源主导全局语境 => 权重更高（竞争可解释）。"""
    ws = Workspace({"perc": 16, "goal": 16}, broadcast_dim=32, seed=5)
    weak = np.full(16, 0.01)
    strong = np.full(16, 100.0)
    _, w_strong_perc = ws(perc=strong, goal=weak)
    _, w_strong_goal = ws(perc=weak, goal=strong)
    assert w_strong_perc["perc"] > w_strong_perc["goal"]
    assert w_strong_goal["goal"] > w_strong_goal["perc"]


def test_temperature_sharpens_competition():
    ws = Workspace(DIMS, broadcast_dim=32, seed=7)
    rng = np.random.default_rng(8)
    args = (rng.normal(size=16), rng.normal(size=16),
            rng.normal(size=4), rng.normal(size=8))
    _, w_hot = ws(*args, temperature=0.01)
    _, w_cold = ws(*args, temperature=100.0)
    # 高温 => 接近均匀；低温 => 更尖锐
    assert max(w_hot.values()) >= max(w_cold.values())
    assert max(w_cold.values()) <= 1.0 / 3 + 0.05  # 四源接近均匀


def test_broadcast_changes_with_inputs():
    ws = Workspace(DIMS, broadcast_dim=32, seed=0)
    rng = np.random.default_rng(9)
    args = (rng.normal(size=16), rng.normal(size=16),
            rng.normal(size=4), rng.normal(size=8))
    b1, _ = ws(*args)
    b2, _ = ws(*args)
    b3, _ = ws(-args[0], args[1], args[2], args[3])
    assert np.allclose(b1, b2)   # 确定性
    assert not np.allclose(b1, b3)  # 输入变化 => 广播变化
