"""M25 边界条件测试。"""
import pytest

from cog_body import Body, BodyError


def test_invalid_constructor_values():
    with pytest.raises(BodyError):
        Body(energy=1.5)
    with pytest.raises(BodyError):
        Body(integrity=-0.1)
    with pytest.raises(BodyError):
        Body(arena_size=0)


def test_invalid_move_type():
    body = Body()
    with pytest.raises(BodyError, match="move"):
        body.step({"move": "north"})
    with pytest.raises(BodyError, match="move"):
        body.step({"move": (1,)})


def test_negative_eat_and_damage_rejected():
    body = Body()
    with pytest.raises(BodyError, match="eat"):
        body.step({"eat": -0.1})
    with pytest.raises(BodyError, match="damage"):
        body.step({"damage": -0.5})


def test_values_clamped():
    body = Body(energy=0.9)
    body.step({"eat": 5.0, "damage": 5.0})
    assert body.energy == 1.0
    assert body.integrity == 0.0


def test_dead_body_returns_zero_effects():
    body = Body(energy=0.0, integrity=0.0)
    body.alive = False
    assert body.step({"move": (1, 1), "eat": 1.0}) == {"energy": 0.0, "integrity": 0.0}


def test_death_when_both_depleted():
    body = Body(energy=0.001, integrity=0.001)
    body.step({"damage": 0.01, "move": (1.0, 0.0)})
    assert not body.alive


def test_tiny_move_negligible_cost():
    body = Body()
    body.step({"move": (1e-9, 1e-9)})
    assert body.energy == pytest.approx(1.0, abs=1e-9)  # 距离过小，能耗可忽略


def test_empty_action_ok():
    body = Body()
    effects = body.step({})
    assert effects == {"energy": 0.0, "integrity": 0.0}
