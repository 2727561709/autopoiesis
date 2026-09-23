"""M25 接口契约测试。"""
import inspect

from cog_body import Body


def test_api_surface():
    for name in ("step", "state"):
        assert hasattr(Body, name)


def test_step_signature():
    sig = inspect.signature(Body.step)
    assert list(sig.parameters)[:2] == ["self", "action"]


def test_step_returns_drive_effects():
    body = Body()
    effects = body.step({})
    assert set(effects) == {"energy", "integrity"}
    assert all(isinstance(v, float) for v in effects.values())


def test_state_contract():
    s = Body().state()
    assert set(s) == {"position", "energy", "integrity", "alive"}
