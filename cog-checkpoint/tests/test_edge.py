"""M03 边界条件测试。"""
import gzip
import pickle

import pytest

from cog_checkpoint import Checkpoint, CheckpointError


def test_empty_state(tmp_path):
    f = tmp_path / "empty.ckpt"
    Checkpoint.save(None, f)
    assert Checkpoint.load(f) is None
    Checkpoint.save({}, f)
    assert Checkpoint.load(f) == {}


def test_unicode_and_large_strings(tmp_path):
    state = {"desc": "认知智能体" * 1000}
    f = tmp_path / "u.ckpt"
    Checkpoint.save(state, f)
    assert Checkpoint.load(f) == state


def test_nested_directory_auto_created(tmp_path):
    f = tmp_path / "deep" / "a" / "b" / "x.ckpt"
    Checkpoint.save({"x": 1}, f)
    assert f.exists()


def test_gzip_magic_byte(tmp_path):
    f = tmp_path / "magic.ckpt"
    Checkpoint.save({"x": 1}, f, compress=True)
    with open(f, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"  # gzip 魔数


def test_truncated_file_raises(tmp_path):
    f = tmp_path / "trunc.ckpt"
    Checkpoint.save({"x": 1}, f)
    data = f.read_bytes()
    f.write_bytes(data[: len(data) // 2])
    with pytest.raises(CheckpointError):
        Checkpoint.load(f)


def test_corrupted_magic_raises(tmp_path):
    f = tmp_path / "corrupt.ckpt"
    # 构造一个合法 gzip + pickle 但 magic 错误的文件
    payload = {"magic": "someone-else", "meta": None, "state": {}}
    with gzip.open(f, "wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(CheckpointError, match="not a valid"):
        Checkpoint.load(f)


def test_latest_with_pattern(tmp_path):
    d = tmp_path / "pats"
    d.mkdir()
    Checkpoint.save({"i": 1}, d / "a.bin", compress=False)
    Checkpoint.save({"i": 2}, d / "b.ckpt")
    assert Checkpoint.latest(d, pattern="*.bin").name == "a.bin"
    assert Checkpoint.latest(d, pattern="*.ckpt").name == "b.ckpt"
