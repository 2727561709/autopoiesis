"""cog-logger: M02 日志模块公开 API。
cog-logger: public API of the M02 logging module.

用法 / usage:
    from cog_logger import Logger
    log = Logger(log_dir="runs", experiment_name="exp1")
    log.info("hello", step=1)
    log.metric("loss", 0.5, step=1)
    log.close()
"""
from .core import Logger
from .types import LogStats

__all__ = ["Logger", "LogStats"]
__version__ = "0.1.0"
