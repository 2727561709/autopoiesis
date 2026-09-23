# cog-world (M26)

世界环境：食物、危险、地形、资源。

## 接口

```python
from cog_world import World

world = World(size=10, n_food=10, n_danger=3, seed=0)
aid = world.add_agent()                        # 或 add_agent(body) 用现成 M25 身体
obs = world.observe(aid)                       # (6,) 归一化观测
obs, reward = world.step(aid, {"move": (0.5, 0.0)})
world.reset(seed)                              # 重新生成布局
```

- 观测 `OBS_DIM=6`：`[x/size, y/size, 食物气味, 危险接近度, 能量, 完整性]`。
- 交互：走到食物 `EAT_RADIUS` 内自动吃掉（+0.3 能量）；进入危险区受 `DANGER_DAMAGE` 伤害。
- 奖励（内在奖励雏形，M16 规划器接入前的过渡）：`Δenergy + Δintegrity - 0.01 存活开销`。
- 智能体即 M25 `Body`，模块间仅通过接口通信。

## 测试

```bash
pip install -e .[dev] && pytest
```
