# cog-drives API

## `DriveSystem(decay=0.995, crisis_threshold=0.15, setpoints=None)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 推进 | `step(effects=None) -> Dict[str, float]` | 衰减 + 效果 + 裁剪 |
| 缺口 | `deficit() -> Dict[str, float]` / `deficit_vector() -> (5,)` | setpoint − value |
| 危机 | `in_crisis() -> List[str]` | 缺口 > 阈值的驱动名 |
| 向量 | `to_tensor() -> (5,)` | 顺序 = `DRIVE_NAMES` |
| 状态 | `state() -> Dict` | 值 + 缺口（可视化/日志用） |
| 重置 | `reset() -> None` | 回到设定点 |

`DRIVE_NAMES = ["energy", "integrity", "curiosity", "social", "competence"]`。

异常：`DriveError(ValueError)` —— 参数越界、未知驱动名/效果键。
