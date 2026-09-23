"""M12 边界条件测试。"""
import pytest

from cog_drives import DRIVE_NAMES, DriveError, DriveSystem


def test_invalid_decay():
    with pytest.raises(DriveError):
        DriveSystem(decay=0.0)
    with pytest.raises(DriveError):
        DriveSystem(decay=1.5)


def test_invalid_crisis_threshold():
    with pytest.raises(DriveError):
        DriveSystem(crisis_threshold=-0.1)
    with pytest.raises(DriveError):
        DriveSystem(crisis_threshold=1.1)


def test_unknown_setpoint_key():
    with pytest.raises(DriveError, match="unknown drive"):
        DriveSystem(setpoints={"hunger": 0.5})


def test_setpoint_out_of_range():
    with pytest.raises(DriveError, match="setpoint"):
        DriveSystem(setpoints={"energy": 1.5})


def test_unknown_effect_key():
    drives = DriveSystem()
    with pytest.raises(DriveError, match="unknown effect"):
        drives.step({"hunger": 0.1})


def test_zero_decay_boundary_ok():
    DriveSystem(decay=1.0)  # 不衰减，合法


def test_long_run_stays_in_bounds():
    drives = DriveSystem(decay=0.999)
    for _ in range(10000):
        drives.step({"energy": 0.002, "integrity": -0.001})
    for d in DRIVE_NAMES:
        assert 0.0 <= drives.values[d] <= 1.0


def test_none_effects_allowed():
    drives = DriveSystem()
    out = drives.step(None)  # 相当于只衰减
    assert set(out) == set(DRIVE_NAMES)
