"""M07 边界与防御测试。"""
import numpy as np
import pytest

from cog_episodic import EpisodicMemory, MemoryError


def test_bad_constructor():
    with pytest.raises(MemoryError):
        EpisodicMemory(capacity=0)
    with pytest.raises(MemoryError):
        EpisodicMemory(latent_dim=-1)


def test_write_shape_and_nan():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    with pytest.raises(MemoryError):
        mem.write(np.ones(5))                   # 维度错
    with pytest.raises(MemoryError):
        mem.write(np.ones((2, 4)))              # 2-D
    z = np.ones(4); z[0] = np.nan
    with pytest.raises(MemoryError):
        mem.write(z)


def test_read_bad_args():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    mem.write(np.ones(4))
    with pytest.raises(MemoryError):
        mem.read(np.ones(4), k=0)
    with pytest.raises(MemoryError):
        mem.read(np.ones(5))                    # 查询维度错
    q = np.ones(4); q[0] = np.inf
    with pytest.raises(MemoryError):
        mem.read(q)


def test_read_empty_memory():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    res = mem.read(np.ones(4), k=3)
    assert not res.episodes and not res.sims and not res


def test_recent_and_sample_bad_args():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    with pytest.raises(MemoryError):
        mem.recent(0)
    with pytest.raises(MemoryError):
        mem.sample(-1)
    with pytest.raises(MemoryError):            # 空记忆不可回放
        mem.sample(4)
    mem.write(np.ones(4))
    assert len(mem.recent(10)) == 1             # n 超过存量不报错
    assert len(mem.sample(10)) == 1


def test_k_over_capacity():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    for i in range(3):
        mem.write(np.full(4, (i + 1) / 4))
    assert len(mem.read(np.ones(4), k=100).episodes) == 3


def test_state_dict_mismatch():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    sd = mem.state_dict()
    mem2 = EpisodicMemory(capacity=10, latent_dim=8)
    with pytest.raises(MemoryError):
        mem2.load_state_dict(sd)


def test_meta_merge():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    ep = mem.write(np.ones(4), {"action": 1}, reward=0.5, done=False)
    assert ep.get("action") == 1 and ep.get("reward") == 0.5
    assert ep.get("done") is False              # 关键字覆盖同名字典键
