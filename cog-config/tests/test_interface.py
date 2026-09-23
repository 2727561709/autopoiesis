"""M01 接口契约测试：保证 API 签名稳定，模块替换时上层不受影响。"""
import inspect

from cog_config import Config


def test_api_surface():
    # 公开的类与函数
    assert callable(Config.load)
    assert callable(Config.validate)
    assert hasattr(Config, "to_dict")
    assert hasattr(Config, "save")


def test_load_signature():
    sig = inspect.signature(Config.load)
    params = list(sig.parameters)
    assert params[0] == "path"  # 位置参数 path
    assert sig.return_annotation is not None


def test_load_returns_config(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("latent_dim: 8\n", encoding="utf-8")
    cfg = Config.load(f)
    assert isinstance(cfg, Config)


def test_dataclass_field_type_stability():
    import dataclasses
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    for key in ("latent_dim", "hidden_dim", "seed", "device"):
        assert key in types
    assert types["latent_dim"] == "int"
    assert types["device"] == "str"
