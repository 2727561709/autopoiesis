# cog-config API

## `Config` (dataclass)

| 方法 | 签名 | 说明 |
|---|---|---|
| 加载 | `Config.load(path=None, **overrides) -> Config` | YAML + env + kwargs 合并 |
| 校验 | `validate() -> None` | 失败抛 `ConfigError` |
| 校验(详细) | `validate_detailed() -> ValidationResult` | 收集全部错误 |
| 导出 | `to_dict() -> dict` | dataclass 转字典 |
| 保存 | `save(path) -> None` | 写 YAML |

字段见 `src/cog_config/core.py` 的 dataclass 定义（约 30 个，覆盖 M02/M03/M06/M07/M12/M16/M24 的公共参数）。

## 异常

- `ConfigError(ValueError)`：文件缺失、YAML 语法错误、根非映射、校验失败。
