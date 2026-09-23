# cog-goal-manager (M15)

目标堆维护，从驱动缺口生成目标。

## 接口

```python
from cog_goal_manager import Goal, GoalManager

gm = GoalManager(max_goals=10, deficit_threshold=0.1)
gm.update_from_drives(drives.deficit())   # 缺口 > 阈值的驱动 -> "satisfy_{drive}"
gm.add(Goal(name="explore", priority=0.3, deadline=100))
goal = gm.current(step=t)                  # 过期处理 + 最高优先级活跃目标
gm.complete(goal.name)
```

- 优先级 ∈ [0,1]，缺口越大优先级越高（只升不降）。
- 同名 add = 更新并重新激活；堆满驱逐优先级最低的活跃目标。
- `current(step)` 自动处理 deadline 过期（expired）。
- 完成的目标可因缺口再次升高而重新激活。

零依赖（纯 Python dataclass），通过字典接口与 M12 解耦。

## 测试

```bash
pip install -e .[dev] && pytest
```
