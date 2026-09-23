"""M06 边界条件测试。"""
import numpy as np
import pytest

from cog_working_memory import WorkingMemory, WorkingMemoryError


def test_read_empty_memory_raises():
    wm = WorkingMemory(capacity=4, dim=3)
    with pytest.raises(WorkingMemoryError, match="empty"):
        wm.read([1.0, 0.0, 0.0])
    with pytest.raises(WorkingMemoryError, match="empty"):
        wm.attention([1.0, 0.0, 0.0])


def test_write_wrong_dim_raises():
    wm = WorkingMemory(capacity=4, dim=3)
    with pytest.raises(WorkingMemoryError, match="expected dim"):
        wm.write(np.ones(5))


def test_write_after_clear_keeps_dim_constraint():
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write(np.ones(3))
    wm.clear()
    with pytest.raises(WorkingMemoryError):
        wm.write(np.ones(4))  # dim 仍锁定为 3


def test_write_nan_inf_raises():
    wm = WorkingMemory(capacity=4, dim=3)
    with pytest.raises(WorkingMemoryError, match="NaN"):
        wm.write([np.nan, 0.0, 0.0])
    with pytest.raises(WorkingMemoryError, match="NaN"):
        wm.write([np.inf, 0.0, 0.0])


def test_write_2d_raises():
    wm = WorkingMemory(capacity=4, dim=3)
    with pytest.raises(WorkingMemoryError, match="1-D"):
        wm.write(np.ones((2, 3)))


def test_invalid_constructor():
    with pytest.raises(WorkingMemoryError):
        WorkingMemory(capacity=0)
    with pytest.raises(WorkingMemoryError):
        WorkingMemory(capacity=4, dim=-1)


def test_zero_query_uniform_ish_attention():
    """零向量的余弦相似度定义为 0 => 权重接近均匀（数值稳定）。"""
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write([1.0, 0.0, 0.0])
    wm.write([0.0, 1.0, 0.0])
    w = wm.attention([0.0, 0.0, 0.0])
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0.0)


def test_zero_slot_does_not_crash():
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write([0.0, 0.0, 0.0])
    wm.write([1.0, 0.0, 0.0])
    w = wm.attention([1.0, 0.0, 0.0])
    assert w.sum() == pytest.approx(1.0)
    assert int(np.argmax(w)) == 1  # 零槽位不相似


def test_capacity_one():
    wm = WorkingMemory(capacity=1, dim=2)
    wm.write([1.0, 0.0])
    wm.write([0.0, 1.0])
    assert len(wm) == 1
    assert np.allclose(wm.read([0.0, 1.0]), [0.0, 1.0])


def test_large_values_stable():
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write([1e8, -1e8, 0.0])
    w = wm.attention([1e8, -1e8, 0.0])
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)
