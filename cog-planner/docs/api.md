# cog-planner API

## `Planner(world_model, action_dim, discrete=True, horizon=8, population=64, elite_frac=0.25, generations=4, goal_weight=1.0, seed=0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 规划 | `plan(z, goal=None) -> PlanResult` | CEM 想象搜索最优动作序列 |
| 重置 | `reset()` | 重置连续模式分布（换任务时） |

`world_model`：鸭子类型，只需 `forward(z, a) -> (z', r, done)`
（M10 `WorldModel` 直接可用）。

目标函数：`score = Σ r_t - goal_weight * ||z_H - goal||₂`
（goal 为 None 时纯回报最大化）。

`PlanResult`：`actions (H, A)`、`zs`（H+1）、`rewards`、`dones`、
`score`、`history`（每代最优）、`first_action`（MPC 单步执行）。

异常：`PlannerError(ValueError)` —— 非法构造参数、z/goal 维度不匹配、
NaN/Inf。
