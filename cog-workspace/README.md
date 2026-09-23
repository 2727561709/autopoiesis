# cog-workspace (M14)

多源竞争性广播，决定注意焦点（全局工作空间理论的向量化近似）。

## 接口

```python
from cog_workspace import Workspace

ws = Workspace(source_dims={"perc": 128, "mem": 128, "emo": 8, "goal": 32},
               broadcast_dim=64, seed=0)
broadcast, weights = ws(perc=z, mem=z_mem, emo=e, goal=g, temperature=1.0)
# weights 例: {"perc": 0.42, "mem": 0.31, "emo": 0.05, "goal": 0.22}
```

- 各源经随机投影 + tanh 映射到广播空间（同种子可复现）。
- 竞争：全局语境 `comp = tanh(Σ hᵢ)`，得分 `⟨hᵢ, comp⟩/√d`；与语境越一致、
  越显著的源权重越高。
- 广播 = 权重加权和；权重和为 1、按源命名，可解释、可直接记录。
- `None` 的源不参与本步竞争；至少要有一个激活源。

## 测试

```bash
pip install -e .[dev] && pytest
```
