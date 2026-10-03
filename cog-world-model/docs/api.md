# cog-world-model API

## `WorldModel(latent_dim, action_dim, hidden_dim=128, seed=0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 预测 | `forward(z, a) / __call__(z, a)` | `(z,a)`->`(z', r, done)`；单个或批量 |
| 记录 | `observe(z, a, z_next, reward=0, done=0)` | 存入回放缓冲（r 截断到 [-1,1]） |
| 训练 | `train_step(lr, batch_size=32) -> dict` | 采样 SGD，返回三头 MSE 损失 |
| 好奇 | `prediction_error(z, a, z_next) -> float` | 预测误差（内在奖励接口） |
| 想象 | `imagine(z, policy, horizon) -> ImagineResult` | 脑内 rollout，done>0.5 提前终止 |
| 存取 | `state_dict() / load_state_dict(sd)` | 纯 NumPy 数组 + 缓冲，可 pickle |

动作类型：`int`（自动 one-hot）、`(action_dim,)` 向量、int 列表
（批量离散）、`(B, action_dim)` 矩阵。

`ImagineResult`：`zs`（H+1）、`actions`、`rewards`、`dones`、
`total_reward`、`terminated`。

异常：`WorldModelError(ValueError)` —— 维度不匹配、越界动作、
NaN/Inf、空缓冲训练、负 horizon、state_dict 形状不符。
