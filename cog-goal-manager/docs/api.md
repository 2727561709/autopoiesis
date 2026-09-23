# cog-goal-manager API

## `GoalManager(max_goals=10, deficit_threshold=0.1)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 添加 | `add(goal, priority=0.5, deadline=None) -> Goal` | 同名更新并重激活 |
| 驱动生成 | `update_from_drives(deficits) -> List[Goal]` | 缺口>阈值 => `satisfy_{drive}`，优先级=缺口 |
| 当前目标 | `current(step=None) -> Goal \| None` | 过期处理 + 最高优先级 |
| 完成 | `complete(name) -> bool` | |
| 过期 | `expire(step) -> List[str]` | 返回过期目标名 |
| 查询 | `active() -> List[Goal]` / `clear()` | 活跃目标按优先级降序 |

## `Goal` dataclass

`name: str`、`priority: float ∈ [0,1]`、`deadline: int | None`、
`status: active/completed/expired`、`metadata: dict`。

异常：`GoalManagerError(ValueError)`。
