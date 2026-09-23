# cog-workspace API

## `Workspace(source_dims, broadcast_dim=64, seed=0)`

`source_dims`: `{"perc": d, "mem": d, "emo": d, "goal": d}` 的非空子集，
键必须属于 `("perc", "mem", "emo", "goal")`。

| 方法 | 签名 | 说明 |
|---|---|---|
| 前向 | `forward(perc=None, mem=None, emo=None, goal=None, temperature=1.0)` | 返回 `(broadcast, weights)` |

- `broadcast`: `(broadcast_dim,)`，各源 tanh 隐表示的凸组合，有界 `(-1,1)`。
- `weights`: `{"perc": w, ...}`，仅含激活源，和为 1。

异常：`WorkspaceError(ValueError)` —— 无激活源、源未配置、维度不匹配、
NaN/Inf、非法温度/构造参数。
