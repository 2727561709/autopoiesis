"""M03 核心功能测试：存取一致性、版本兼容、压缩。"""
import gzip
import pickle
import sys

import pytest

from cog_checkpoint import Checkpoint, CheckpointError, CheckpointMeta, FORMAT_VERSION


def test_roundtrip_dict(tmp_path):
    state = {"step": 42, "weights": [1.0, 2.0, 3.0], "name": "agent"}
    f = tmp_path / "a.ckpt"
    meta = Checkpoint.save(state, f)
    assert isinstance(meta, CheckpointMeta)
    loaded = Checkpoint.load(f)
    assert loaded == state


def test_roundtrip_nested(tmp_path):
    state = {"memory": {"episodic": [1, 2], "semantic": {"a": 1}},
             "tuple": (1, "x"), "none": None}
    f = tmp_path / "b.ckpt"
    Checkpoint.save(state, f)
    assert Checkpoint.load(f) == state


def test_meta_stored(tmp_path):
    f = tmp_path / "c.ckpt"
    Checkpoint.save({"s": 1}, f, module_versions={"cog-config": "0.1.0"}, note="first")
    payload = Checkpoint.load_with_meta(f)
    meta = payload["meta"]
    assert meta.format_version == FORMAT_VERSION
    assert meta.module_versions == {"cog-config": "0.1.0"}
    assert meta.note == "first"
    assert meta.created_at > 0


def test_compression_effective(tmp_path):
    state = {"big": list(range(50000))}
    fc = tmp_path / "compressed.ckpt"
    fu = tmp_path / "raw.ckpt"
    Checkpoint.save(state, fc, compress=True)
    Checkpoint.save(state, fu, compress=False)
    # 重复整数序列压缩后应明显更小
    assert fc.stat().st_size < fu.stat().st_size
    assert Checkpoint.load(fc) == Checkpoint.load(fu) == state


def test_atomic_no_tmp_left(tmp_path):
    f = tmp_path / "atomic.ckpt"
    Checkpoint.save({"x": 1}, f)
    Checkpoint.save({"x": 2}, f)  # 覆盖写
    assert not (tmp_path / "atomic.ckpt.tmp").exists()
    assert Checkpoint.load(f)["x"] == 2


def test_latest(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    import time
    f1 = d / "step_10.ckpt"
    Checkpoint.save({"i": 10}, f1)
    time.sleep(0.05)
    f2 = d / "step_20.ckpt"
    Checkpoint.save({"i": 20}, f2)
    assert Checkpoint.latest(d) == f2
    assert Checkpoint.latest(tmp_path / "empty_dir") is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        Checkpoint.load(tmp_path / "nope.ckpt")


def test_version_incompatibility(tmp_path):
    f = tmp_path / "future.ckpt"
    Checkpoint.save({"x": 1}, f)
    with pytest.raises(CheckpointError, match="newer"):
        Checkpoint.load(f, expect_version=0)


def test_not_a_checkpoint(tmp_path):
    f = tmp_path / "garbage.ckpt"
    f.write_bytes(pickle.dumps({"something": "else"}))
    with pytest.raises(CheckpointError, match="not a valid"):
        Checkpoint.load(f)


def test_unpicklable_state_raises(tmp_path):
    with pytest.raises(CheckpointError, match="failed to save"):
        Checkpoint.save({"bad": lambda x: x}, tmp_path / "bad.ckpt")
    assert not (tmp_path / "bad.ckpt").exists()
    assert not (tmp_path / "bad.ckpt.tmp").exists()
