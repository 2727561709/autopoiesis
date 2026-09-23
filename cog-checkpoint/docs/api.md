# cog-checkpoint API

## `Checkpoint`（静态方法）

| 方法 | 签名 | 说明 |
|---|---|---|
| 保存 | `save(state, path, module_versions=None, note=None, compress=True) -> CheckpointMeta` | 原子写入 |
| 加载 | `load(path, expect_version=None) -> Any` | 返回 state |
| 加载+元数据 | `load_with_meta(path, expect_version=None) -> dict` | `{"meta", "state"}` |
| 最新存档 | `latest(directory, pattern="*.ckpt") -> Optional[Path]` | 按 mtime |

存档文件格式：`gzip(pickle({"magic": "cog-checkpoint", "meta": CheckpointMeta, "state": Any}))`。

`CheckpointMeta`：`format_version`（当前 1）、`created_at`、`module_versions: Dict[str,str]`、`note`。

异常：`CheckpointError(RuntimeError)`。
