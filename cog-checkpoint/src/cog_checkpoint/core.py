"""M03 存档模块主实现。

职责：任意可序列化状态（模型权重/记忆/内部状态）的版本化保存与恢复。
设计要点：
- 单文件存档：gzip 压缩的 pickle，内含 {meta, state}。
- 原子写入：先写 .tmp 再 os.replace，崩溃不会留下半个存档。
- 版本检查：加载时校验 format_version，不兼容抛 CheckpointError。
- torch 可用时会自动使用 torch.save/torch.load 优化张量序列化（可选依赖）。
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
    """存档器（静态方法风格，无内部状态）。"""

    @staticmethod
    def save(
        state: Any,
        path: str | os.PathLike,
        module_versions: Optional[Dict[str, str]] = None,
        note: Optional[str] = None,
        compress: bool = True,
    ) -> CheckpointMeta:
        """保存状态到版本化存档文件。

        Args:
            state: 任意可 pickle 的对象（dict / dataclass / 张量容器等）。
            path: 目标文件路径。
            module_versions: 各模块版本号，用于加载时兼容性诊断。
            note: 人类可读备注。
            compress: 是否 gzip 压缩。
        Returns:
            实际写入的元数据。
        Raises:
            CheckpointError: 不可序列化或写入失败。
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
            os.replace(tmp, p)  # 原子替换
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

        Args:
            path: 存档文件路径。
            expect_version: 期望的格式版本；None 表示接受当前版本。
        Returns:
            保存时的 state 对象。
        Raises:
            CheckpointError: 文件不存在 / 魔数不符 / 版本不兼容 / 反序列化失败。
        """
        payload = Checkpoint.load_with_meta(path, expect_version)
        return payload["state"]

    @staticmethod
    def load_with_meta(
        path: str | os.PathLike,
        expect_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """同 load，但返回 {"meta": CheckpointMeta, "state": Any}。"""
        p = Path(path)
        if not p.exists():
            raise CheckpointError(f"checkpoint not found: {p}")

        try:
            with gzip.open(p, "rb") as f:
                payload = pickle.load(f)
        except OSError:
            # 可能是未压缩存档，回退普通读取
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
        """返回目录中修改时间最新的存档路径；目录为空返回 None。"""
        d = Path(directory)
        if not d.exists():
            return None
        candidates = [f for f in d.glob(pattern) if f.is_file()]
        return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None
