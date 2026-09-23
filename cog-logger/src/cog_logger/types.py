"""M02 日志模块的输入输出类型定义（接口契约）。
M02 logging module — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union

#: 一条结构化事件就是任意可 JSON 序列化的映射
#: A structured event is any JSON-serializable mapping
Event = Mapping[str, Any]
#: 指标值：标量即可 / a metric value: any scalar
MetricValue = Union[int, float]


@dataclass
class LogStats:
    """日志器运行统计（用于测试与监控）。
    Logger runtime statistics (for tests and monitoring)."""

    events_written: int = 0
    metrics_written: int = 0
    dropped: int = 0  # 队列满时丢弃的条数 / items dropped when the queue was full
