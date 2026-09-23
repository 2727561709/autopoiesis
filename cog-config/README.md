# cog-config (M01)

统一管理认知智能体所有超参数、路径、开关。

## 接口

```python
from cog_config import Config

cfg = Config.load("config.yaml")   # YAML + 环境变量(COG_前缀) + kwargs 覆盖
cfg.validate()                     # 校验失败抛 ConfigError
cfg.to_dict() / cfg.save(path)     # 导出
```

覆盖优先级：默认值 < YAML < 环境变量 < kwargs。

## 环境变量

`COG_<FIELD>=value`，自动做 bool/int/float 类型转换，如 `COG_LATENT_DIM=256`、`COG_LOG_ASYNC=false`。未知键与未知环境变量静默忽略（向后兼容）。

## 测试

```bash
pip install -e .[dev] && pytest
```
