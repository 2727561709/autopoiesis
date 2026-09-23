"""M01 配置模块主实现。

职责：统一管理所有超参数、路径、开关。
来源优先级（低 -> 高）：内置默认值 < YAML 文件 < 环境变量（COG_ 前缀）。
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from .types import ConfigError, ValidationResult

#: 环境变量前缀，例如 COG_LATENT_DIM=256 覆盖 latent_dim
ENV_PREFIX = "COG_"


@dataclass
class Config:
    # ---- 全局 ----
    seed: int = 42
    device: str = "cpu"
    latent_dim: int = 128
    hidden_dim: int = 256

    # ---- 日志（与 M02 对接）----
    experiment_name: str = "cog-agent"
    log_dir: str = "runs"
    log_level: str = "INFO"
    log_async: bool = True

    # ---- 存档（与 M03 对接）----
    checkpoint_dir: str = "checkpoints"
    checkpoint_compress: bool = True

    # ---- 工作记忆 M06 ----
    working_memory_capacity: int = 16

    # ---- 情景记忆 M07 ----
    episodic_capacity: int = 10000
    recall_topk: int = 5

    # ---- 驱动 M12 ----
    drive_decay: float = 0.995
    drive_crisis_threshold: float = 0.15

    # ---- 规划器 M16 ----
    plan_horizon: int = 8
    cem_samples: int = 256
    cem_iterations: int = 4

    # ---- 主循环 M24 ----
    max_steps: int = 100000

    def __post_init__(self) -> None:
        """构造即校验，保证任何途径得到的 Config 都是合法的。"""
        self.validate()

    # ------------------------------------------------------------------ API
    @classmethod
    def load(cls, path: str | os.PathLike | None = None, **overrides: Any) -> "Config":
        """从 YAML 文件加载配置，支持环境变量与关键字参数覆盖。

        Args:
            path: YAML 文件路径，None 表示仅使用默认值 + 环境变量。
            **overrides: 直接的关键字覆盖（最高优先级）。
        Returns:
            类型安全的 Config 对象（已通过 validate）。
        Raises:
            ConfigError: 文件不存在 / YAML 语法错误 / 校验失败。
        """
        file_values: Dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if not p.exists():
                raise ConfigError(f"config file not found: {p}")
            try:
                loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                raise ConfigError(f"invalid YAML in {p}: {e}") from e
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"YAML root must be a mapping, got {type(loaded).__name__}")
            file_values = dict(loaded)

        merged: Dict[str, Any] = dict(file_values)
        merged.update(cls._from_env())
        merged.update(overrides)

        return cls._from_filtered(merged)

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """校验配置合法性，失败抛出 ConfigError。"""
        result = self.validate_detailed()
        if not result.ok:
            raise ConfigError("; ".join(result.errors))

    def validate_detailed(self) -> ValidationResult:
        errors: List[str] = []
        positive_ints = [
            "latent_dim", "hidden_dim", "working_memory_capacity",
            "episodic_capacity", "recall_topk", "plan_horizon",
            "cem_samples", "cem_iterations", "max_steps", "seed",
        ]
        for name in positive_ints:
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                errors.append(f"{name} must be a positive int, got {v!r}")

        if not (0.0 < self.drive_decay <= 1.0):
            errors.append(f"drive_decay must be in (0, 1], got {self.drive_decay!r}")
        if not (0.0 <= self.drive_crisis_threshold <= 1.0):
            errors.append(f"drive_crisis_threshold must be in [0, 1], got {self.drive_crisis_threshold!r}")
        if self.device not in ("cpu", "cuda", "mps"):
            errors.append(f"device must be one of cpu/cuda/mps, got {self.device!r}")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            errors.append(f"invalid log_level: {self.log_level!r}")
        return ValidationResult(ok=not errors, errors=errors)

    # ------------------------------------------------------------------ 工具
    def to_dict(self) -> Dict[str, Any]:
        """导出为普通字典（用于存档与日志）。"""
        return asdict(self)

    def save(self, path: str | os.PathLike) -> None:
        """把当前配置写回 YAML（方便实验复现）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ 内部
    @classmethod
    def _from_env(cls) -> Dict[str, Any]:
        """收集 COG_ 前缀的环境变量并做类型转换。"""
        out: Dict[str, Any] = {}
        field_types = {f.name: f.type for f in fields(cls)}
        for name, value in os.environ.items():
            if not name.startswith(ENV_PREFIX):
                continue
            key = name[len(ENV_PREFIX):].lower()
            if key not in field_types:
                continue
            out[key] = _coerce(key, value)
        return out

    @classmethod
    def _from_filtered(cls, values: Mapping[str, Any]) -> "Config":
        """忽略未知键（保持向后兼容），构造并校验。"""
        valid = {f.name for f in fields(cls)}
        unknown = [k for k in values if k not in valid]
        # 未知键静默忽略：旧配置文件在模块升级后仍可加载
        kwargs = {k: v for k, v in values.items() if k in valid}
        cfg = cls(**kwargs)
        try:
            cfg.validate()
        except ConfigError as e:
            detail = f" (unknown keys ignored: {unknown})" if unknown else ""
            raise ConfigError(f"{e}{detail}") from None
        return cfg


def _coerce(name: str, raw: str) -> Any:
    """按 'true/false' -> bool，'数字' -> int/float 的顺序尝试转换。"""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
