"""M01 核心功能测试：加载、覆盖、校验、默认值。"""
import os

import pytest

from cog_config import Config, ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.latent_dim == 128
    assert cfg.device == "cpu"
    assert cfg.seed == 42


def test_yaml_load(tmp_path):
    f = tmp_path / "cfg.yaml"
    f.write_text("latent_dim: 64\ndevice: cuda\n", encoding="utf-8")
    cfg = Config.load(f)
    assert cfg.latent_dim == 64
    assert cfg.device == "cuda"
    # 未指定的字段使用默认值
    assert cfg.hidden_dim == 256


def test_yaml_empty_and_null(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    assert Config.load(f).latent_dim == 128
    f2 = tmp_path / "null.yaml"
    f2.write_text("# only comment\n", encoding="utf-8")
    assert Config.load(f2).latent_dim == 128


def test_env_override(tmp_path, monkeypatch):
    f = tmp_path / "cfg.yaml"
    f.write_text("latent_dim: 64\n", encoding="utf-8")
    monkeypatch.setenv("COG_LATENT_DIM", "256")
    monkeypatch.setenv("COG_LOG_ASYNC", "false")
    cfg = Config.load(f)
    assert cfg.latent_dim == 256
    assert cfg.log_async is False  # 环境变量 bool 转换


def test_kwargs_override_all(tmp_path, monkeypatch):
    f = tmp_path / "cfg.yaml"
    f.write_text("latent_dim: 64\n", encoding="utf-8")
    monkeypatch.setenv("COG_LATENT_DIM", "256")
    cfg = Config.load(f, latent_dim=512)  # kwargs 最高优先级
    assert cfg.latent_dim == 512


def test_validate_raises_on_bad_value():
    with pytest.raises(ConfigError, match="latent_dim"):
        Config(latent_dim=-1)


def test_validate_detailed_collects_all_errors():
    cfg = Config()
    # 绕过构造校验，直接注入非法值以测试错误收集
    cfg.latent_dim = 0
    cfg.drive_decay = 2.0
    cfg.device = "tpu"
    result = cfg.validate_detailed()
    assert not result.ok
    assert len(result.errors) == 3


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "nope.yaml")


def test_bad_yaml_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        Config.load(f)


def test_non_mapping_root_raises(tmp_path):
    f = tmp_path / "list.yaml"
    f.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        Config.load(f)


def test_unknown_keys_ignored(tmp_path):
    f = tmp_path / "future.yaml"
    f.write_text("latent_dim: 32\nthis_field_does_not_exist: 1\n", encoding="utf-8")
    cfg = Config.load(f)  # 不抛异常，向后兼容
    assert cfg.latent_dim == 32


def test_to_dict_and_roundtrip(tmp_path):
    cfg = Config(latent_dim=99)
    d = cfg.to_dict()
    assert d["latent_dim"] == 99
    f = tmp_path / "dump.yaml"
    cfg.save(f)
    cfg2 = Config.load(f)
    assert cfg2 == cfg
