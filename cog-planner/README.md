# cog-planner (M16)

基于 M10 世界模型的 CEM/MPC 想象规划器。

## 接口

```python
from cog_planner import Planner

pl = Planner(world_model,            # 任何 forward(z, a) -> (z', r, done) 兼容对象
             action_dim=5, discrete=True,
             horizon=8, population=64, elite_frac=0.25, generations=4,
             goal_weight=1.0, seed=0)

res = pl.plan(z)                      # 纯回报最大化
res = pl.plan(z, goal=z_goal)         # 回报 + 终局接近目标

res.actions       # (H, action_dim) 最优动作序列（离散 = one-hot）
res.zs            # 预测轨迹，长度 H+1
res.score         # Σr - goal_weight * ||z_H - goal||
res.history       # 每代 CEM 最优得分（收敛轨迹）
res.first_action  # MPC 风格：只执行第一步，其余留待重规划
```

- **CEM**：采样 H 步动作序列 → 世界模型想象评估 → 精英拟合 → 迭代。
- 离散模式：每步类别分布 + 拉普拉斯平滑（防过早坍缩）；
  连续模式：每步高斯，裁剪到 [-1, 1]，`reset()` 换任务后重置。
- 想象中预测 done > 0.5 时提前终止，剩余步补 0 回报。

## 测试

```bash
pip install -e .[dev] && pytest
```
