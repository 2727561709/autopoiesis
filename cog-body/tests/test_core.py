"""M25 核心功能测试：能量守恒、损伤、运动。"""
import math

import pytest

from cog_body import MOVE_COST_PER_UNIT, Body


def test_initial_state():
    body = Body(position=(2.0, 3.0))
    s = body.state()
    assert s["position"] == (2.0, 3.0)
    assert s["energy"] == 1.0
    assert s["integrity"] == 1.0
    assert s["alive"]


def test_move_costs_energy_and_updates_position():
    body = Body(position=(0.0, 0.0), energy=1.0)
    effects = body.step({"move": (3.0, 4.0)})  # 距离 5
    assert body.position == (3.0, 4.0)
    assert effects["energy"] == pytest.approx(-5 * MOVE_COST_PER_UNIT)
    assert body.energy == pytest.approx(1.0 - 5 * MOVE_COST_PER_UNIT)


def test_energy_conservation():
    """能量变化 == 内部能量差（守恒一致性）。"""
    body = Body(energy=0.8)
    e0 = body.energy
    effects = body.step({"move": (1.0, 1.0), "eat": 0.1})
    assert body.energy == pytest.approx(e0 + effects["energy"])


def test_damage_reduces_integrity():
    body = Body()
    effects = body.step({"damage": 0.3})
    assert body.integrity == pytest.approx(0.7)
    assert effects["integrity"] == pytest.approx(-0.3)


def test_eat_increases_energy():
    body = Body(energy=0.5)
    body.step({"eat": 0.2})
    assert body.energy == pytest.approx(0.7)


def test_rest_recovers_energy():
    body = Body(energy=0.5)
    effects = body.step({"rest": True})
    assert effects["energy"] > 0
    assert body.energy > 0.5


def test_position_clamped_to_arena():
    body = Body(position=(9.9, 0.1), arena_size=10.0)
    body.step({"move": (5.0, -5.0)})
    assert body.position == (10.0, 0.0)


def test_move_with_no_energy_shortened():
    """能量不足时按剩余能量等比缩短位移。"""
    body = Body(energy=0.01)
    body.step({"move": (10.0, 0.0)})
    assert body.energy == pytest.approx(0.0)          # 能量耗尽
    assert body.position[0] == pytest.approx(0.01 / MOVE_COST_PER_UNIT)


def test_combined_actions():
    body = Body(position=(0.0, 0.0), energy=0.5)
    effects = body.step({"move": (1.0, 0.0), "eat": 0.3, "damage": 0.2})
    assert effects["energy"] == pytest.approx(0.3 - MOVE_COST_PER_UNIT)
    assert effects["integrity"] == pytest.approx(-0.2)
    assert body.position == (1.0, 0.0)


def test_zero_move_no_cost():
    body = Body()
    effects = body.step({"move": (0.0, 0.0)})
    assert effects["energy"] == 0.0
    assert body.position == (0.0, 0.0)
