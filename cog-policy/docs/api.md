# cog-policy API

## `Policy(latent_dim, action_dim, hidden_dim=64, seed=0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 前向 | `forward(z) / __call__(z) -> (DiagGaussian, ndarray)` | dist + V(z) |
| 动作 | `act(z, rng, deterministic=False) -> (action, log_prob, value)` | 单步交互 |
| 反传 | `backward(grad_mean, grad_value, grad_log_std=None)` | 外部算好对输出的梯度 |
| 训练 | `zero_grad()` / `sgd_step(lr)` / `parameters()` | 与 torch 命名对齐 |
| 存取 | `state_dict()` / `load_state_dict(sd)` | 纯 NumPy 数组 |

## `DiagGaussian(mean, std)`

`sample(rng)`、`log_prob(action)`（各维求和）、`entropy()`、`deterministic()`。

mean 由 tanh 网络给出（有界 `[-1,1]`）；std 由可学习 log_std 限幅 `exp(clip(log_std, -5, 2))`。

异常：`PolicyError(ValueError)`。
