# cog-working-memory (M06)

容量有限的短期缓存，注意力读写。

## 接口

```python
from cog_working_memory import WorkingMemory

wm = WorkingMemory(capacity=16, dim=128)
wm.write(z)                  # 写入（满则 FIFO 驱逐最旧）
z_out = wm.read(query)       # 余弦注意力加权聚合
w = wm.attention(query)      # 权重（和为 1，可解释）
wm.clear(); len(wm); wm.state()
```

- 槽位：滑动窗口，FIFO 驱逐。
- 读取：`softmax(cos(query, slot_i))`，零向量相似度定义为 0（数值稳定）。
- `dim` 可在构造时指定，或由首次写入推断（之后锁定）。

## 测试

```bash
pip install -e .[dev] && pytest
```
