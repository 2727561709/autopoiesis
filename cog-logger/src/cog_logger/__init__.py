"""cog-logger: M02 日志模块公开 API。

用法:
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
