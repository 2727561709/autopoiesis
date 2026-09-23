"""M12 接口契约测试。"""
import inspect

import numpy as np

from cog_drives import DRIVE_NAMES, DriveSystem


def test_api_surface():
    for name in ("step", "deficit", "deficit_vector", "in_crisis", "to_tensor",
                 "state", "reset"):
        assert hasattr(DriveSystem, name)


def test_step_signature():
    sig = inspect.signature(DriveSystem.step)
    assert list(sig.parameters)[:2] == ["self", "effects"]


def test_step_returns_dict_of_all_drives():
    drives = DriveSystem()
    out = drives.step()
    assert set(out) == set(DRIVE_NAMES)


def test_to_tensor_contract():
    t = DriveSystem().to_tensor()
    assert isinstance(t, np.ndarray) and t.shape == (5,)


def test_drive_name_order_stable():
    assert DRIVE_NAMES == ["energy", "integrity", "curiosity", "social", "competence"]
