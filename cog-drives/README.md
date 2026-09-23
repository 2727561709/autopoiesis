# cog-drives (M12)

稳态驱动系统：能量、完整性、好奇、社交、胜任。

## 接口

```python
from cog_drives import DriveSystem

drives = DriveSystem(decay=0.995, crisis_threshold=0.15)
drives.step({"energy": +0.3})      # 衰减 + 动作效果
gap = drives.deficit()              # {"energy": 0.12, ...}
vec = drives.to_tensor()            # (5,) 顺序固定
crisis = drives.in_crisis()         # ["energy"]
```

模型：值 v ∈ [0,1]，每步 `v = clip(v * decay + effect, 0, 1)`；缺口 = 设定点 − 值（裁剪）。设定点默认 1.0（好奇等可通过 `setpoints` 调低）。

与 M25 对接：`drives.step(body.step(action))` —— 身体动作的效果直接作为驱动输入。

## 测试

```bash
pip install -e .[dev] && pytest
```
