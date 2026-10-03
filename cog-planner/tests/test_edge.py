"""M16 边界与防御测试。"""
import numpy as np
import pytest

from cog_planner import Planner, PlannerError

from stub_wm import StubWM


def test_bad_constructor():
    wm = StubWM()
    with pytest.raises(PlannerError):
        Planner(wm, action_dim=0)
    with pytest.raises(PlannerError):
        Planner(wm, action_dim=3, horizon=0)
    with pytest.raises(PlannerError):
        Planner(wm, action_dim=3, population=1)
    with pytest.raises(PlannerError):
        Planner(wm, action_dim=3, elite_frac=0.0)
    with pytest.raises(PlannerError):
        Planner(wm, action_dim=3, elite_frac=1.5)
    with pytest.raises(PlannerError):
        Planner(wm, action_dim=3, generations=0)


def test_bad_z():
    pl = Planner(StubWM(), action_dim=3, horizon=3, population=8,
                 generations=1)
    with pytest.raises(PlannerError):
        pl.plan(np.ones((2, 4)))                 # 2-D
    with pytest.raises(PlannerError):
        z = np.ones(4); z[0] = np.nan
        pl.plan(z)


def test_goal_dim_mismatch():
    pl = Planner(StubWM(), action_dim=3, horizon=3, population=8,
                 generations=1)
    with pytest.raises(PlannerError):
        pl.plan(np.ones(4), goal=np.ones(5))
    with pytest.raises(PlannerError):
        pl.plan(np.ones(4), goal=np.ones((2, 4)))


def test_minimal_population():
    """population=2、elite=1 也能跑（不崩即可）。population=2 with
    elite=1 runs without crashing."""
    pl = Planner(StubWM(), action_dim=3, horizon=2, population=2,
                 generations=2, seed=0)
    res = pl.plan(np.ones(4))
    assert res.actions.shape == (2, 3)


def test_onehot_validity_discrete():
    pl = Planner(StubWM(action_dim=3), action_dim=3, horizon=4,
                 population=16, generations=2)
    res = pl.plan(np.zeros(4))
    assert np.allclose(res.actions.sum(axis=-1), 1.0)   # 每步合法 one-hot
    assert set(np.unique(res.actions)) <= {0.0, 1.0}


def test_reset():
    pl = Planner(StubWM(), action_dim=3, discrete=False, horizon=3,
                 population=8, generations=2)
    pl.plan(np.zeros(4))
    pl.reset()
    assert np.all(pl._mean == 0.0) and np.all(pl._std == 1.0)
