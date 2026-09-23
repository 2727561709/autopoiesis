"""M03 接口契约测试。"""
import inspect

from cog_checkpoint import Checkpoint, CheckpointError, CheckpointMeta


def test_api_surface():
    for name in ("save", "load", "load_with_meta", "latest"):
        assert hasattr(Checkpoint, name)


def test_signatures():
    save_params = list(inspect.signature(Checkpoint.save).parameters)
    assert save_params[:2] == ["state", "path"]
    load_params = list(inspect.signature(Checkpoint.load).parameters)
    assert load_params[:2] == ["path", "expect_version"]


def test_save_returns_meta_load_returns_state(tmp_path):
    f = tmp_path / "contract.ckpt"
    meta = Checkpoint.save(state={"k": "v"}, path=f)
    assert isinstance(meta, CheckpointMeta)
    state = Checkpoint.load(f)
    assert state == {"k": "v"}


def test_exception_type():
    assert issubclass(CheckpointError, RuntimeError)
