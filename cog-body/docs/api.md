# cog-body API

## `Body(position=(0,0), energy=1.0, integrity=1.0, arena_size=10.0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 推进 | `step(action) -> {"energy": Δ, "integrity": Δ}` | 效果直接给 M12 |
| 状态 | `state() -> Dict` | position/energy/integrity/alive |

动作键：`move (dx,dy)`、`eat float`、`rest bool`、`damage float`（可组合）。

常量：`MOVE_COST_PER_UNIT = 0.02`、`REST_RECOVERY = 0.005`。

异常：`BodyError(ValueError)` —— 参数越界、动作格式非法。
