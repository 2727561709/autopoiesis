"""M06 核心功能测试：容量、读写、注意力权重。"""
import numpy as np
import pytest

from cog_working_memory import WorkingMemory, WorkingMemoryError


def test_write_read_basic():
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write([1.0, 0.0, 0.0])
    out = wm.read([1.0, 0.0, 0.0])
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 0.0, 0.0])  # 唯一条目 => 原样读出


def test_exact_match_dominates():
    wm = WorkingMemory(capacity=8, dim=3)
    wm.write([1.0, 0.0, 0.0])
    wm.write([0.0, 1.0, 0.0])
    wm.write([0.0, 0.0, 1.0])
    out = wm.read([0.0, 1.0, 0.0])
    # 读出应是三个槽的凸组合，且以第二个为主
    assert out[1] > out[0] and out[1] > out[2]


def test_attention_weights_sum_to_one():
    wm = WorkingMemory(capacity=8, dim=5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        wm.write(rng.normal(size=5))
    w = wm.attention(rng.normal(size=5))
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0.0)


def test_attention_peaks_at_most_similar_slot():
    wm = WorkingMemory(capacity=8, dim=4)
    slots = [np.eye(4)[i] for i in range(4)]
    for s in slots:
        wm.write(s)
    w = wm.attention([0.0, 0.0, 1.0, 0.0])
    assert int(np.argmax(w)) == 2


def test_capacity_fifo_eviction():
    wm = WorkingMemory(capacity=3, dim=2)
    for i in range(6):
        wm.write([float(i), 1.0])
    assert len(wm) == 3
    # 只剩最后 3 条：3, 4, 5
    w = wm.attention([5.0, 1.0])
    assert int(np.argmax(w)) == 2  # 最新条目在末尾
    w0 = wm.attention([0.0, 1.0])
    # 最早的条目已被驱逐，剩余三条与查询相似度接近 => 权重接近均匀
    assert np.allclose(w0, [1 / 3] * 3, atol=0.05)


def test_read_returns_convex_combination():
    wm = WorkingMemory(capacity=4, dim=3)
    wm.write([1.0, 1.0, 1.0])
    wm.write([-1.0, -1.0, -1.0])
    out = wm.read([0.1, 0.0, 0.0])
    assert np.all(out <= 1.0 + 1e-9) and np.all(out >= -1.0 - 1e-9)


def test_repeated_write_same_vector():
    wm = WorkingMemory(capacity=2, dim=2)
    wm.write([1.0, 0.0])
    wm.write([1.0, 0.0])
    assert len(wm) == 2
    assert np.allclose(wm.read([1.0, 0.0]), [1.0, 0.0])


def test_clear():
    wm = WorkingMemory(capacity=4, dim=2)
    wm.write([1.0, 0.0])
    wm.clear()
    assert len(wm) == 0
    with pytest.raises(WorkingMemoryError):
        wm.read([1.0, 0.0])


def test_dim_inferred_from_first_write():
    wm = WorkingMemory(capacity=4)
    assert wm.dim is None
    wm.write(np.ones(7))
    assert wm.dim == 7


def test_state_snapshot():
    wm = WorkingMemory(capacity=5, dim=3)
    wm.write(np.zeros(3))
    s = wm.state()
    assert s == {"size": 1, "capacity": 5, "dim": 3}
