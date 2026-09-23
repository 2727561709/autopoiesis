"""M12 核心功能测试：衰减、恢复、危机判定。"""
import numpy as np
import pytest

from cog_drives import DRIVE_NAMES, DriveError, DriveSystem


def test_initial_values_at_setpoints():
    drives = DriveSystem()
    assert drives.values == {d: 1.0 for d in DRIVE_NAMES}


def test_decay_over_time():
    drives = DriveSystem(decay=0.99)
    for _ in range(100):
        drives.step()
    assert drives.values["energy"] == pytest.approx(0.99 ** 100, rel=1e-6)
    assert drives.values["energy"] < 1.0


def test_effects_recovery():
    drives = DriveSystem(decay=1.0)
    drives.values["energy"] = 0.5
    drives.step({"energy": +0.3})
    assert drives.values["energy"] == pytest.approx(0.8)


def test_values_clamped_to_unit_interval():
    drives = DriveSystem(decay=1.0)
    drives.step({"energy": +5.0, "integrity": -5.0})
    assert drives.values["energy"] == 1.0
    assert drives.values["integrity"] == 0.0


def test_deficit():
    drives = DriveSystem(decay=1.0)
    drives.values["energy"] = 0.6
    d = drives.deficit()
    assert d["energy"] == pytest.approx(0.4)
    assert d["integrity"] == 0.0


def test_deficit_never_negative():
    drives = DriveSystem(decay=1.0, setpoints={"curiosity": 0.5})
    drives.values["curiosity"] = 0.9  # 超过设定点
    assert drives.deficit()["curiosity"] == 0.0


def test_crisis_detection():
    drives = DriveSystem(decay=1.0, crisis_threshold=0.15)
    drives.values["energy"] = 0.7     # 缺口 0.3 > 0.15
    drives.values["integrity"] = 0.9  # 缺口 0.1 < 0.15
    assert drives.in_crisis() == ["energy"]


def test_to_tensor():
    drives = DriveSystem()
    t = drives.to_tensor()
    assert isinstance(t, np.ndarray)
    assert t.shape == (5,)
    assert list(t) == [1.0] * 5


def test_deficit_vector_shape():
    drives = DriveSystem()
    assert drives.deficit_vector().shape == (5,)
    assert float(drives.deficit_vector().sum()) == 0.0


def test_reset():
    drives = DriveSystem(decay=0.9)
    for _ in range(50):
        drives.step()
    assert drives.values["energy"] < 1.0
    drives.reset()
    assert drives.values["energy"] == 1.0


def test_state_contains_values_and_deficits():
    drives = DriveSystem(decay=1.0)
    drives.values["social"] = 0.5
    s = drives.state()
    assert s["social"] == 0.5
    assert s["social_deficit"] == 0.5
