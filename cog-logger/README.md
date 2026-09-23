# cog-logger (M02)

结构化日志、指标记录、实验追踪。

## 接口

```python
from cog_logger import Logger

log = Logger(log_dir="runs", experiment_name="exp1", async_mode=True)
log.info("msg", **fields)          # 结构化事件 -> exp1.events.jsonl
log.metric("loss", 0.5, step=1)    # 标量指标   -> exp1.metrics.jsonl
log.flush(); log.close()
```

- 异步模式：后台守护线程写 JSONL，主循环零阻塞；队列满则丢弃并计入 `stats.dropped`。
- 控制台同时输出摘要（可 `console=False` 关闭）。
- 可选 TensorBoard：安装 `tensorboard` 后自动记录 scalar（`pip install .[tb]`）。
- 与 M01 对接：`Logger(log_dir=cfg.log_dir, experiment_name=cfg.experiment_name, level=cfg.log_level, async_mode=cfg.log_async)`。

## 测试

```bash
pip install -e .[dev] && pytest
```
