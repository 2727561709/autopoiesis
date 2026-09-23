"""M06 接口契约测试。"""
import inspect

import numpy as np

from cog_working_memory import WorkingMemory


def test_api_surface():
    for name in ("write", "read", "attention", "clear", "state"):
        assert hasattr(WorkingMemory, name)


def test_signatures():
    assert list(inspect.signature(WorkingMemory.write).parameters) == ["self", "z"]
    assert list(inspect.signature(WorkingMemory.read).parameters) == ["self", "query"]


def test_write_read_contract():
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write(np.ones(3))
    out = wm.read(np.ones(3))
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_len_protocol():
    wm = WorkingMemory(capacity=4, dim=2)
    assert len(wm) == 0
    wm.write([1.0, 0.0])
    assert len(wm) == 1
