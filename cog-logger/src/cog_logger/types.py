"""M02 日志模块的输入输出类型定义（接口契约）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union

#: 一条结构化事件就是任意可 JSON 序列化的映射
Event = Mapping[str, Any]
#: 指标值：标量即可
MetricValue = Union[int, float]


@dataclass
class LogStats:
    """日志器运行统计（用于测试与监控）。"""

    events_written: int = 0
    metrics_written: int = 0
    dropped: int = 0  # 队列满时丢弃的条数
