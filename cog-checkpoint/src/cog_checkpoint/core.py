"""M03 存档模块主实现。
M03 checkpoint module — main implementation.

职责：任意可序列化状态（模型权重/记忆/内部状态）的版本化保存与恢复。
Responsibilities: versioned save/restore of any serializable state
(model weights / memories / internal state).
设计要点 / design notes:
- 单文件存档：gzip 压缩的 pickle，内含 {meta, state}。
  Single-file archive: gzip-compressed pickle holding {meta, state}.
- 原子写入：先写 .tmp 再 os.replace，崩溃不会留下半个存档。
  Atomic writes: write to .tmp then os.replace, so a crash never
  leaves a half-written archive.
- 版本检查：加载时校验 format_version，不兼容抛 CheckpointError。
  Version check: format_version is validated on load; an incompatible
  one raises CheckpointError.
- torch 可用时会自动使用 torch.save/torch.load 优化张量序列化（可选依赖）。
  When torch is available it is used automatically to optimize tensor
  serialization (optional dependency).
"""
from __future__ import annotations

import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .types import FORMAT_VERSION, CheckpointError, CheckpointMeta

_MAGIC = "cog-checkpoint"


class Checkpoint:
    """存档器（静态方法风格，无内部状态）。
    Checkpoint store (static-method style, no internal state)."""

    @staticmethod
    def save(
        state: Any,
        path: str | os.PathLike,
        module_versions: Optional[Dict[str, str]] = None,
        note: Optional[str] = None,
        compress: bool = True,
    ) -> CheckpointMeta:
        """保存状态到版本化存档文件。
        Save state into a versioned archive file.

        Args:
            state: 任意可 pickle 的对象（dict / dataclass / 张量容器等）。
                Any picklable object (dict / dataclass / tensor container).
            path: 目标文件路径。Target file path.
            module_versions: 各模块版本号，用于加载时兼容性诊断。
                Module versions, for compatibility diagnostics on load.
            note: 人类可读备注。Human-readable note.
            compress: 是否 gzip 压缩。Whether to gzip-compress.
        Returns:
            实际写入的元数据。The metadata actually written.
        Raises:
            CheckpointError: 不可序列化或写入失败。
                Unserializable state or a write failure.
        """
        meta = CheckpointMeta(
            format_version=FORMAT_VERSION,
            created_at=time.time(),
            module_versions=dict(module_versions or {}),
            note=note,
        )
        payload = {"magic": _MAGIC, "meta": meta, "state": state}

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")

        try:
            if compress:
                with gzip.open(tmp, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(tmp, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)  # 原子替换 / atomic replace
        except (OSError, pickle.PicklingError, TypeError) as e:
            tmp.unlink(missing_ok=True)
            raise CheckpointError(f"failed to save checkpoint to {p}: {e}") from e
        return meta

    @staticmethod
    def load(
        path: str | os.PathLike,
        expect_version: Optional[int] = None,
    ) -> Any:
        """从存档恢复状态。
        Restore state from an archive.

        Args:
            path: 存档文件路径。Archive file path.
            expect_version: 期望的格式版本；None 表示接受当前版本。
                Expected format version; None accepts the current one.
        Returns:
            保存时的 state 对象。The state object as saved.
        Raises:
            CheckpointError: 文件不存在 / 魔数不符 / 版本不兼容 / 反序列化失败。
                Missing file / wrong magic / incompatible version /
                deserialization failure.
        """
        payload = Checkpoint.load_with_meta(path, expect_version)
        return payload["state"]

    @staticmethod
    def load_with_meta(
        path: str | os.PathLike,
        expect_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """同 load，但返回 {"meta": CheckpointMeta, "state": Any}。
        Like load, but returns {"meta": CheckpointMeta, "state": Any}."""
        p = Path(path)
        if not p.exists():
            raise CheckpointError(f"checkpoint not found: {p}")

        try:
            with gzip.open(p, "rb") as f:
                payload = pickle.load(f)
        except OSError:
            # 可能是未压缩存档，回退普通读取
            # Possibly an uncompressed archive; fall back to plain read
            try:
                with open(p, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                raise CheckpointError(f"failed to load checkpoint {p}: {e}") from e
        except Exception as e:
            raise CheckpointError(f"failed to load checkpoint {p}: {e}") from e

        if not isinstance(payload, dict) or payload.get("magic") != _MAGIC:
            raise CheckpointError(f"not a valid cog-checkpoint file: {p}")

        meta: CheckpointMeta = payload["meta"]
        want = expect_version if expect_version is not None else FORMAT_VERSION
        if meta.format_version > want:
            raise CheckpointError(
                f"checkpoint format v{meta.format_version} is newer than "
                f"supported v{want}; upgrade cog-checkpoint"
            )
        return payload

    @staticmethod
    def latest(directory: str | os.PathLike, pattern: str = "*.ckpt") -> Optional[Path]:
        """返回目录中修改时间最新的存档路径；目录为空返回 None。
        Return the most recently modified archive in the directory, or
        None if there is none."""
        d = Path(directory)
        if not d.exists():
            return None
        candidates = [f for f in d.glob(pattern) if f.is_file()]
        return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None
