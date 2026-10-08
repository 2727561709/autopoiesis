# cog-semantic-memory (M08)

知识图谱语义记忆：概念 + 带类型加权关系，用**扩散激活**做关联检索。

## 接口

```python
from cog_semantic import SemanticMemory

mem = SemanticMemory(decay=0.5, threshold=0.05)

# 建图
mem.add_concept("hungry", kind="drive")
mem.add_concept("food"); mem.add_concept("berry"); mem.add_concept("bush")
mem.link("hungry", "food", "motivates", 1.0)
mem.link("food", "berry", "is-a", 0.8)
mem.link("berry", "bush", "grows-on", 0.9)
mem.link("pond", "bush", "near", 0.6, bidirectional=True)

# 扩散激活："此刻哪些概念被唤起？"（按激活值降序）
act = mem.spread(["hungry"], depth=3)
# {'hungry': 1.0, 'food': 0.5, 'berry': 0.2, 'bush': ...}

act = mem.spread({"hungry": 1.0, "thirsty": 0.6})   # 多种子（可加权）

# 图查询
mem.get("food"); mem.neighbors("hungry"); "food" in mem; len(mem)

mem.state_dict() / mem.load_state_dict(sd)
```

- 传播规则：`激活_next = Σ 激活_src × weight × decay`，逐跳衰减，
  低于 `threshold` 的概念被剔除；激活只增不减（重复命中取最大）。
- 与 M07 情景记忆分工：M07 存**经历**（向量，相似检索）；本模块存
  **知识**（符号，图检索）。

## 测试

```bash
pip install -e .[dev] && pytest
```
