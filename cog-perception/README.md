# cog-perception (M04)

把原始观测编码成潜向量。

## 接口

```python
from cog_perception import Encoder

enc = Encoder(obs_dim=16, latent_dim=128, hidden_dim=256, seed=0)
z = enc(obs)                     # (128,)  值域 (-1, 1)
zb = enc(batch)                  # (B, 128)
zm = enc({"vision": v, "audio": a})  # 多模态拼接

# 训练（与 torch 对齐的命名）
enc.zero_grad(); enc.backward(grad_z); enc.sgd_step(lr)
enc.state_dict() / enc.load_state_dict(sd)
```

- 后端：NumPy 微型网络（`nn.py`：Linear/Tanh/Sequential + 手写反向传播），无 torch 依赖。
- 输出 tanh 有界化，保证下游数值稳定。
- 接口与 torch 命名对齐，未来切换 torch 后端时上层代码零改动。

## 测试

```bash
pip install -e .[dev] && pytest
```
