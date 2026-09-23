"""M01 配置模块主实现。
M01 configuration module — main implementation.

职责：统一管理所有超参数、路径、开关。
Responsibilities: centrally manage all hyperparameters, paths, and
switches.
来源优先级（低 -> 高）：内置默认值 < YAML 文件 < 环境变量（COG_ 前缀）。
Source priority (low -> high): built-in defaults < YAML file <
environment variables (COG_ prefix).
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
#: Environment-variable prefix, e.g. COG_LATENT_DIM=256 overrides latent_dim
ENV_PREFIX = "COG_"


@dataclass
class Config:
    """全局配置（构造即校验）。Global config (validated on construction)."""

    # ---- 全局 / global ----
    seed: int = 42
    device: str = "cpu"
    latent_dim: int = 128
    hidden_dim: int = 256

    # ---- 日志（与 M02 对接）/ logging (feeds M02) ----
    experiment_name: str = "cog-agent"
    log_dir: str = "runs"
    log_level: str = "INFO"
    log_async: bool = True

    # ---- 存档（与 M03 对接）/ checkpoints (feeds M03) ----
    checkpoint_dir: str = "checkpoints"
    checkpoint_compress: bool = True

    # ---- 工作记忆 M06 / working memory M06 ----
    working_memory_capacity: int = 16

    # ---- 情景记忆 M07 / episodic memory M07 ----
    episodic_capacity: int = 10000
    recall_topk: int = 5

    # ---- 驱动 M12 / drives M12 ----
    drive_decay: float = 0.995
    drive_crisis_threshold: float = 0.15

    # ---- 规划器 M16 / planner M16 ----
    plan_horizon: int = 8
    cem_samples: int = 256
    cem_iterations: int = 4

    # ---- 主循环 M24 / main loop M24 ----
    max_steps: int = 100000

    def __post_init__(self) -> None:
        """构造即校验，保证任何途径得到的 Config 都是合法的。
        Validate on construction, so every Config instance is legal
        however it was obtained."""
        self.validate()

    # ------------------------------------------------------------------ API
    @classmethod
    def load(cls, path: str | os.PathLike | None = None, **overrides: Any) -> "Config":
        """从 YAML 文件加载配置，支持环境变量与关键字参数覆盖。
        Load config from a YAML file, with env-var and keyword
        overrides.

        Args:
            path: YAML 文件路径，None 表示仅使用默认值 + 环境变量。
                YAML file path; None means defaults + env vars only.
            **overrides: 直接的关键字覆盖（最高优先级）。
                Direct keyword overrides (highest priority).
        Returns:
            类型安全的 Config 对象（已通过 validate）。
                A type-safe Config object (already validated).
        Raises:
            ConfigError: 文件不存在 / YAML 语法错误 / 校验失败。
                Missing file / YAML syntax error / validation failure.
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

    # ------------------------------------------------------------ validation
    def validate(self) -> None:
        """校验配置合法性，失败抛出 ConfigError。
        Validate the config; raises ConfigError on failure."""
        result = self.validate_detailed()
        if not result.ok:
            raise ConfigError("; ".join(result.errors))

    def validate_detailed(self) -> ValidationResult:
        """校验并返回结构化结果（不抛异常）。
        Validate and return a structured result (no exception)."""
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

    # --------------------------------------------------------------- helpers
    def to_dict(self) -> Dict[str, Any]:
        """导出为普通字典（用于存档与日志）。
        Export to a plain dict (for checkpointing and logging)."""
        return asdict(self)

    def save(self, path: str | os.PathLike) -> None:
        """把当前配置写回 YAML（方便实验复现）。
        Write the current config back to YAML (for experiment
        reproducibility)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    # -------------------------------------------------------------- internals
    @classmethod
    def _from_env(cls) -> Dict[str, Any]:
        """收集 COG_ 前缀的环境变量并做类型转换。
        Collect COG_-prefixed env vars with type coercion."""
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
        """忽略未知键（保持向后兼容），构造并校验。
        Ignore unknown keys (backward compatible); construct and
        validate."""
        valid = {f.name for f in fields(cls)}
        unknown = [k for k in values if k not in valid]
        # 未知键静默忽略：旧配置文件在模块升级后仍可加载
        # Unknown keys are silently ignored so old config files keep
        # loading after module upgrades.
        kwargs = {k: v for k, v in values.items() if k in valid}
        cfg = cls(**kwargs)
        try:
            cfg.validate()
        except ConfigError as e:
            detail = f" (unknown keys ignored: {unknown})" if unknown else ""
            raise ConfigError(f"{e}{detail}") from None
        return cfg


def _coerce(name: str, raw: str) -> Any:
    """按 'true/false' -> bool，'数字' -> int/float 的顺序尝试转换。
    Try 'true/false' -> bool, then 'number' -> int/float, in that
    order."""
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
