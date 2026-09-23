# cog-world API

## `World(size=10.0, n_food=10, n_danger=3, seed=0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 重置 | `reset(seed=None) -> None` | 重新生成食物/危险布局 |
| 注册 | `add_agent(body=None, position=None) -> int` | 返回递增 agent_id |
| 观测 | `observe(agent_id) -> (OBS_DIM,)` | 归一化局部观测 |
| 交互 | `step(agent_id, action) -> (obs, reward)` | 身体动作 + 环境效果 |

观测布局：`[x/size, y/size, food_scent, danger_proximity, energy, integrity]`。

常量：`OBS_DIM=6`、`EAT_RADIUS=0.5`、`DANGER_RADIUS=0.5`、`DANGER_DAMAGE=0.1`。

依赖：`cog-body`（M25）。异常：`WorldError`。
