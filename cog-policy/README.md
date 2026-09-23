# cog-policy (M05)

从潜状态生成动作分布 + 价值估计。

## 接口

```python
from cog_policy import Policy

pol = Policy(latent_dim=128, action_dim=4, hidden_dim=64, seed=0)
dist, value = pol(z)                    # DiagGaussian + V(z)
action, log_prob, value = pol.act(z, rng, deterministic=False)
```

- 对角高斯策略：mean 由 MLP（tanh 有界）给出，log_std 可学习并限幅 `[-5, 2]`。
- `DiagGaussian`：`sample / log_prob / entropy / deterministic`，与 torch.distributions 常用子集对齐。
- 价值头独立输出 `V(z)`。
- NumPy 微型网络后端（`nn.py`），未来统一切换 torch。

## 测试

```bash
pip install -e .[dev] && pytest
```
