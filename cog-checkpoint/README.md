# cog-checkpoint (M03)

模型权重、记忆、状态的保存与恢复。

## 接口

```python
from cog_checkpoint import Checkpoint

meta = Checkpoint.save(state, "ckpt/agent.ckpt", module_versions={"cog-config": "0.1.0"})
state = Checkpoint.load("ckpt/agent.ckpt")            # 只取状态
payload = Checkpoint.load_with_meta("ckpt/agent.ckpt")  # {"meta", "state"}
latest = Checkpoint.latest("ckpt/")                   # 最新的 *.ckpt
```

- 单文件 gzip+pickle 存档，内含 `{magic, meta, state}`。
- 原子写入（`.tmp` + `os.replace`），崩溃不留半个文件。
- 版本前向兼容检查：存档格式比代码新时拒绝加载并抛 `CheckpointError`。
- torch 为可选依赖：张量本身可被 pickle，安装 torch 后无需任何代码改动。

## 测试

```bash
pip install -e .[dev] && pytest
```
