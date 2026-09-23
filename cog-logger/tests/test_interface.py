"""M02 接口契约测试。"""
import inspect

from cog_logger import Logger, LogStats


def test_api_surface():
    for name in ("log", "metric", "flush", "close"):
        assert hasattr(Logger, name)


def test_log_signature():
    sig = inspect.signature(Logger.log)
    assert list(sig.parameters)[:2] == ["self", "event"]
    assert list(inspect.signature(Logger.metric).parameters)[:2] == ["self", "name"]


def test_constructor_accepts_dict_like_config():
    # 与 M01 Config 的字段名对齐：log_dir / experiment_name / log_level
    cfg = {"log_dir": "runs", "experiment_name": "e", "log_level": "DEBUG"}
    log = Logger(
        log_dir=cfg["log_dir"],
        experiment_name=cfg["experiment_name"],
        level=cfg["log_level"],
        async_mode=False,
        console=False,
    )
    log.close()
    assert isinstance(log.stats, LogStats)
