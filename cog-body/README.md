# cog-body (M25)

虚拟身体：能量、完整性、位置、行动器。

## 接口

```python
from cog_body import Body

body = Body(position=(1, 1), energy=1.0, integrity=1.0, arena_size=10)
effects = body.step({"move": (0.5, 0.0), "eat": 0.2, "damage": 0.1, "rest": True})
# effects -> {"energy": Δ, "integrity": Δ}   直接喂给 M12: drives.step(effects)
s = body.state()  # {"position", "energy", "integrity", "alive"}
```

- 移动能耗与距离成正比（`MOVE_COST_PER_UNIT`）；能量不足时位移等比缩短。
- 位置限制在 `[0, arena_size]` 竞技场内。
- 能量与完整性同时归零则 `alive = False`。

## 测试

```bash
pip install -e .[dev] && pytest
```
