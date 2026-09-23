"""M01 边界条件测试。"""
import pytest

from cog_config import Config, ConfigError
from cog_config.core import _coerce


def test_minimal_valid_values():
    cfg = Config(latent_dim=1, hidden_dim=1, recall_topk=1, cem_samples=1)
    cfg.validate()  # 不抛异常


def test_boundary_drive_decay():
    Config(drive_decay=1.0).validate()   # 上边界合法
    with pytest.raises(ConfigError):
        Config(drive_decay=0.0)          # 下边界非法
    with pytest.raises(ConfigError):
        Config(drive_decay=1.5)


def test_boundary_crisis_threshold():
    Config(drive_crisis_threshold=0.0).validate()
    Config(drive_crisis_threshold=1.0).validate()
    with pytest.raises(ConfigError):
        Config(drive_crisis_threshold=-0.1)
    with pytest.raises(ConfigError):
        Config(drive_crisis_threshold=1.1)


def test_unicode_experiment_name():
    cfg = Config(experiment_name="认知智能体-实验α")
    assert cfg.experiment_name == "认知智能体-实验α"


def test_env_coercion_edge(monkeypatch):
    monkeypatch.setenv("COG_PLAN_HORIZON", "  16  ")
    monkeypatch.setenv("COG_DRIVE_DECAY", "0.9")
    monkeypatch.setenv("COG_EXPERIMENT_NAME", "hello world")
    cfg = Config.load()
    assert cfg.plan_horizon == 16
    assert cfg.drive_decay == 0.9
    assert cfg.experiment_name == "hello world"


def test_float_where_int_expected(monkeypatch):
    # 环境变量给 float 形式但字段需要 int -> 校验失败
    monkeypatch.setenv("COG_LATENT_DIM", "12.5")
    with pytest.raises(ConfigError, match="latent_dim"):
        Config.load()


def test_unknown_env_vars_ignored(monkeypatch):
    monkeypatch.setenv("COG_TOTALLY_UNKNOWN", "1")
    Config.load()  # 不抛异常


def test_load_with_none_path():
    assert Config.load() is not None
    assert Config.load(None).latent_dim == 128
