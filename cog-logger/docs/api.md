# cog-logger API

## `Logger`

| 方法 | 签名 | 说明 |
|---|---|---|
| 事件 | `log(event: Mapping, level="INFO") -> None` | 结构化事件写 JSONL |
| 指标 | `metric(name: str, value: float, step=None) -> None` | 标量指标写 JSONL (+TB) |
| 级别 | `debug/info/warning/error(msg, **kw)` | `log` 的便捷封装 |
| 落盘 | `flush() -> None` | 等待队列消费完 |
| 关闭 | `close() -> None` | 停线程 + 落盘，幂等 |

属性：`events_path`、`metrics_path`、`stats: LogStats(events_written, metrics_written, dropped)`。

事件格式：`{"ts", "exp", "level", "event": {...}}`；指标格式：`{"ts", "exp", "name", "value", "step"}`。
不可序列化字段自动降级为 `repr` 字符串，保证写入永不失败。
