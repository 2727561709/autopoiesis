"""M07 核心行为测试：相似检索 / 近因 / 回放 / 淘汰 / 存取。"""
import numpy as np

from cog_episodic import EpisodicMemory


def test_similarity_retrieval():
    """查询应命中方向最近的经历。A query should hit the most
    directionally similar experience."""
    mem = EpisodicMemory(capacity=100, latent_dim=8)
    base = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    mem.write(base, tag="east")                 # 与查询同向
    mem.write(-base, tag="west")                # 反向
    mem.write(np.roll(base, 1), tag="north")    # 正交
    res = mem.read(base, k=3)
    assert res.episodes[0].get("tag") == "east"
    assert res.episodes[0].get("tag") != "west"
    assert res.sims[0] > 0.99
    assert res.sims[1] < 0.01                   # north 正交 ~ 0


def test_tie_break_recency():
    """同相似度时按写入顺序的稳定检索。Stable retrieval order for
    equal similarities."""
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    for i in range(5):
        mem.write(np.ones(4), i=i)
    res = mem.read(np.ones(4), k=5)
    assert [e.get("i") for e in res.episodes] == [0, 1, 2, 3, 4]


def test_recent_order():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    for i in range(6):
        mem.write(np.full(4, i / 6), i=i)
    rec = mem.recent(3)
    assert [e.get("i") for e in rec] == [5, 4, 3]   # 新到旧


def test_sample_replay():
    mem = EpisodicMemory(capacity=10, latent_dim=4, seed=0)
    for i in range(10):
        mem.write(np.zeros(4), i=i)
    batch = mem.sample(6)
    assert len(batch) == 6
    assert all(0 <= e.get("i") < 10 for e in batch)
    # 同种子可复现 / same seed reproducible
    mem2 = EpisodicMemory(capacity=10, latent_dim=4, seed=0)
    for i in range(10):
        mem2.write(np.zeros(4), i=i)
    assert [e.get("i") for e in mem2.sample(6)] == [e.get("i") for e in batch]


def test_fifo_eviction():
    mem = EpisodicMemory(capacity=3, latent_dim=4)
    for i in range(5):
        mem.write(np.full(4, i + 1.0), i=i)
    assert len(mem) == 3
    tags = [e.get("i") for e in mem.recent(3)]
    assert tags == [4, 3, 2]                    # 最旧的 0、1 被淘汰


def test_zero_query_similarity():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    mem.write(np.ones(4))
    res = mem.read(np.zeros(4))
    assert res.sims[0] == 0.0                   # 零向量 => 相似度 0


def test_state_dict_roundtrip():
    mem = EpisodicMemory(capacity=50, latent_dim=4)
    for i in range(4):
        mem.write(np.full(4, (i + 1) / 5), action=i, reward=0.1 * i)
    sd = mem.state_dict()
    mem2 = EpisodicMemory(capacity=50, latent_dim=4)
    mem2.load_state_dict(sd)
    assert len(mem2) == 4
    q = np.full(4, 0.6)
    assert [e.t for e in mem.read(q).episodes] == \
           [e.t for e in mem2.read(q).episodes]
    assert mem2.recent(1)[0].get("action") == 3


def test_clock_monotonic_after_clear():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    mem.write(np.ones(4))
    mem.clear()
    assert len(mem) == 0
    ep = mem.write(np.ones(4))
    assert ep.t == 1                            # 清空不回拨时间钟
