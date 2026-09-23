"""cog-config: M01 配置模块公开 API。

用法:
    from cog_config import Config
    cfg = Config.load("config.yaml")
"""
from .core import Config, ENV_PREFIX
from .types import ConfigError, ValidationResult

__all__ = ["Config", "ConfigError", "ValidationResult", "ENV_PREFIX"]
__version__ = "0.1.0"
