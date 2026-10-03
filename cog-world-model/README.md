# cog-world-model (M10)

潜空间世界模型：从 (z, a) 预测 (z', r, done)，并可"脑内"想象 rollout。

## 接口

```python
from cog_world_model import WorldModel

wm = WorldModel(latent_dim=128, action_dim=5, hidden_dim=128, seed=0)

# 预测：单个或批量
z_next, r, done = wm(z, a)        # int 动作自动 one-hot；批量用 [0,1,2..] 或 (B,A)

# 训练（回放缓冲 + 手写反向传播）
wm.observe(z, a, z_next, reward=r, done=d)   # 记录真实转移
loss = wm.train_step(lr=0.05, batch_size=32)  # {"trans","reward","done"}

# 预测误差（exp-a "好奇"内在奖励接口）
err = wm.prediction_error(z, a, z_next)

# 想象 rollout（供 M16 planner 脑内试行）
res = wm.imagine(z, policy=lambda zz: 0, horizon=10)
res.zs            # 长度 H+1 的潜状态序列
res.total_reward  # 累计预测回报
res.terminated    # 预测 done > 0.5 时提前终止
```

- 三个独立头：`trans_net`（z'，tanh 有界）、`reward_net`（r，tanh 有界）、
  `done_net`（(tanh+1)/2 ∈ (0,1)）。
- 后端：NumPy 微型网络（`nn.py`：Linear/Tanh/Sequential + 手写反向传播），
  接口与 torch 命名对齐，未来切换后端零改动。
- `observe()` 里的 reward 会被截断到 [-1, 1]（与 tanh 头匹配）。

## 测试

```bash
pip install -e .[dev] && pytest
```
