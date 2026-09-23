"""M02 日志模块主实现。
M02 logging module — main implementation.

职责：结构化日志（JSONL）、指标记录、实验追踪。
Responsibilities: structured logging (JSONL), metric recording,
experiment tracking.
设计要点 / design notes:
- 异步写入：后台线程消费队列，主循环零阻塞（可关闭）。
  Async writes: a background thread drains a queue so the main loop
  never blocks (can be disabled).
- 标准输出走 logging，结构化事件落 events.jsonl，指标落 metrics.jsonl。
  Human-readable output goes through logging; structured events land
  in events.jsonl and metrics in metrics.jsonl.
- 可选 TensorBoard 后端（安装了 tensorboard 才启用）。
  Optional TensorBoard backend (enabled only when tensorboard is
  installed).
"""
from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .types import LogStats

_SENTINEL = object()


class Logger:
    """结构化日志器。
    Structured logger.

    Args:
        log_dir: 输出目录（自动创建）。Output directory (auto-created).
        experiment_name: 实验名，作为输出前缀。
            Experiment name, used as the output prefix.
        level: 标准日志级别（DEBUG/INFO/WARNING/ERROR）。
            Standard log level (DEBUG/INFO/WARNING/ERROR).
        async_mode: True 时后台线程写入；False 时同步写（测试用）。
            True for background-thread writes; False for synchronous
            writes (used in tests).
        console: 是否同时输出到 stdout。
            Whether to also print to stdout.
        queue_size: 异步队列上限，满则丢弃并计数。
            Async queue cap; overflow items are dropped and counted.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: str = "cog-agent",
        level: str = "INFO",
        async_mode: bool = True,
        console: bool = True,
        queue_size: int = 10000,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self._console = console
        self._stats = LogStats()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.log_dir / f"{experiment_name}.events.jsonl"
        self._metrics_path = self.log_dir / f"{experiment_name}.metrics.jsonl"

        self._py_logger = logging.getLogger(f"cog_logger.{experiment_name}.{id(self)}")
        self._py_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._py_logger.propagate = False
        if console and not self._py_logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._py_logger.addHandler(h)

        # 可选 TensorBoard 后端 / optional TensorBoard backend
        self._tb = None
        try:  # pragma: no cover - 取决于环境 / environment-dependent
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self._tb = SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            pass

        self._async = async_mode
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        if self._async:
            self._thread = threading.Thread(
                target=self._worker, args=(self._queue,),
                name="cog-logger-writer", daemon=True,
            )
            self._thread.start()

        self.info("logger started", log_dir=str(self.log_dir))

    # ------------------------------------------------------------------ API
    def log(self, event: Mapping[str, Any], level: str = "INFO") -> None:
        """记录一条结构化事件。event 必须是可 JSON 序序列化的映射。
        Log one structured event. `event` must be a JSON-serializable
        mapping."""
        record = {
            "ts": time.time(),
            "exp": self.experiment_name,
            "level": level.upper(),
            "event": _safe(event),
        }
        self._emit(("event", record), record)

    def metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """记录一个标量指标。Log one scalar metric."""
        record = {
            "ts": time.time(),
            "exp": self.experiment_name,
            "name": name,
            "value": float(value),
            "step": step if step is not None else int(time.time()),
        }
        if self._tb is not None:  # pragma: no cover
            self._tb.add_scalar(name, record["value"], record["step"])
        self._emit(("metric", record), record)

    # 便捷级别 / convenience levels
    def debug(self, msg: str, **kw: Any) -> None:
        self.log({"msg": msg, **kw}, level="DEBUG")

    def info(self, msg: str, **kw: Any) -> None:
        self.log({"msg": msg, **kw}, level="INFO")

    def warning(self, msg: str, **kw: Any) -> None:
        self.log({"msg": msg, **kw}, level="WARNING")

    def error(self, msg: str, **kw: Any) -> None:
        self.log({"msg": msg, **kw}, level="ERROR")

    def flush(self) -> None:
        """等待异步队列全部落盘（同步模式下为空操作）。
        Wait until the async queue is fully drained (a no-op in
        synchronous mode)."""
        if self._async:
            self._queue.join()

    def close(self) -> None:
        """停止后台线程并落盘。幂等。
        Stop the background thread and flush. Idempotent."""
        if self._async and self._thread.is_alive():
            self._queue.put(_SENTINEL)
            self._thread.join(timeout=10)
            self._thread = threading.Thread(target=lambda: None)  # 防止重复 join / prevents double join
        if self._tb is not None:  # pragma: no cover
            self._tb.close()

    @property
    def stats(self) -> LogStats:
        return self._stats

    @property
    def events_path(self) -> Path:
        return self._events_path

    @property
    def metrics_path(self) -> Path:
        return self._metrics_path

    # -------------------------------------------------------------- internals
    def _emit(self, item: Any, record: Mapping[str, Any]) -> None:
        if self._async:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._stats.dropped += 1
        else:
            self._write(item)
        # 控制台摘要 / console summary
        if self._console:
            msg = record.get("event", {}).get("msg") or record.get("name", "")
            self._py_logger.log(
                getattr(logging, str(record.get("level", "INFO")), logging.INFO), str(msg)
            )

    def _worker(self, q: "queue.Queue[Any]") -> None:
        """消费线程。绑定启动时的队列引用，避免运行中替换导致计数失衡。
        Consumer thread. Bound to the queue captured at start so a
        mid-run swap cannot skew the join counter."""
        while True:
            item = q.get()
            try:
                if item is _SENTINEL:
                    break
                self._write(item)
            finally:
                q.task_done()

    def _write(self, item: Any) -> None:
        kind, record = item
        path = self._events_path if kind == "event" else self._metrics_path
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if kind == "event":
            self._stats.events_written += 1
        else:
            self._stats.metrics_written += 1


def _safe(obj: Any) -> Any:
    """递归地把不可序列化的叶子降级为 repr 字符串，其余保持原样。
    Recursively degrade non-serializable leaves to repr strings,
    leaving everything else untouched."""
    if isinstance(obj, Mapping):
        return {str(k) if not isinstance(k, str) else k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)
