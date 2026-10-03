"""M16 接口契约测试。"""
import inspect

import numpy as np

from cog_planner import PlanResult, Planner, PlannerError

from stub_wm import StubWM


def test_api_surface():
    for name in ("plan", "__call__", "reset"):
        assert hasattr(Planner, name)


def test_plan_signature():
    sig = inspect.signature(Planner.plan)
    assert list(sig.parameters)[:3] == ["self", "z", "goal"]


def test_constructor_signature():
    sig = inspect.signature(Planner.__init__)
    params = list(sig.parameters)
    for expected in ("world_model", "action_dim", "discrete", "horizon",
                     "population", "elite_frac", "generations",
                     "goal_weight", "seed"):
        assert expected in params


def test_result_contract():
    pl = Planner(StubWM(), action_dim=2, horizon=4, population=8,
                 generations=2)
    res = pl.plan([0.1, -0.2, 0.3, -0.4])
    assert isinstance(res, PlanResult)
    assert res.actions.shape == (4, 2)
    assert len(res.rewards) == 4 and len(res.dones) == 4
    assert len(res.zs) == 5                     # H + 1
    assert isinstance(res.score, float)
    assert len(res.history) == 2
    assert isinstance(res.first_action, np.ndarray)


def test_error_type():
    assert issubclass(PlannerError, ValueError)
