# cog-episodic-memory (M07)

长期情景记忆：存储"经历过的情景"，按余弦相似度检索、按近因读取、随机回放。

## 接口

```python
from cog_episodic import EpisodicMemory

mem = EpisodicMemory(capacity=10000, latent_dim=128, seed=0)

# 写入（元数据 = dict 或关键字参数）
ep = mem.write(z, action=0, reward=0.5, done=False)   # ep.t = 经历时间戳

# 检索："这像哪段经历？"（余弦相似 top-k，降序）
res = mem.read(query_z, k=5)
res.episodes / res.sims / res.best

# 近因：最近 n 条（新到旧）
mem.recent(5)

# 回放：随机 n 条（训练数据来源，如喂 M10 世界模型）
mem.sample(32)

len(mem); mem.clear(); mem.state_dict() / mem.load_state_dict(sd)
```

- 与 M06 工作记忆分工：M06 = 短期 FIFO + 注意衰减；本模块 = 长期库 +
  相似检索 + FIFO 容量淘汰（不衰减）。
- 零向量查询相似度定义为 0；`clear()` 不回拨时间钟（t 单调）。

## 测试

```bash
pip install -e .[dev] && pytest
```
