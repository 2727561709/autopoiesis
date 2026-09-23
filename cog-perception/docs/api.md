# cog-perception API

## `Encoder(obs_dim, latent_dim=128, hidden_dim=256, seed=0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 编码 | `forward(obs) / __call__(obs) -> ndarray` | `(obs_dim,)`->`(latent_dim,)` 或 `(B,obs_dim)`->`(B,latent_dim)` |
| 反传 | `backward(grad_z) -> ndarray` | 返回 dL/dobs |
| 训练 | `zero_grad()` / `sgd_step(lr)` / `parameters()` | 与 torch 命名对齐 |
| 存取 | `state_dict()` / `load_state_dict(sd)` | 纯 NumPy 数组，可 pickle |

观测类型：向量、批矩阵、`{name: vector}` 多模态字典（按插入顺序拼接）。

输出：tanh 有界 `(-1, 1)`。

异常：`PerceptionError(ValueError)` —— 维度不匹配、NaN/Inf、空字典、非法构造参数。
