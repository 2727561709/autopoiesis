"""M07 接口契约测试。"""
import inspect

import numpy as np

from cog_episodic import Episode, EpisodicMemory, MemoryError, Retrieved


def test_api_surface():
    for name in ("write", "read", "recent", "sample", "clear",
                 "state_dict", "load_state_dict", "__len__"):
        assert hasattr(EpisodicMemory, name)


def test_write_signature():
    sig = inspect.signature(EpisodicMemory.write)
    params = list(sig.parameters)
    assert params[:3] == ["self", "z", "meta"]


def test_read_signature():
    sig = inspect.signature(EpisodicMemory.read)
    assert list(sig.parameters)[:3] == ["self", "query", "k"]


def test_episode_contract():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    ep = mem.write(np.ones(4), action=0, reward=0.5)
    assert isinstance(ep, Episode)
    assert ep.t == 0
    assert ep.get("action") == 0
    assert ep.get("missing", "dft") == "dft"


def test_retrieved_contract():
    mem = EpisodicMemory(capacity=10, latent_dim=4)
    assert not mem.read(np.ones(4))             # 空记忆 => falsy
    mem.write(np.ones(4))
    res = mem.read(np.ones(4))
    assert isinstance(res, Retrieved)
    assert res.best is res.episodes[0]
    assert bool(res)


def test_error_type():
    assert issubclass(MemoryError, ValueError)
