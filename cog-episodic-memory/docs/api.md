# cog-episodic-memory API

## `EpisodicMemory(capacity=10000, latent_dim=128, seed=0)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 写入 | `write(z, meta=None, **kw_meta) -> Episode` | 返回带时间戳 t 的情景 |
| 检索 | `read(query, k=5) -> Retrieved` | 余弦相似 top-k，降序 |
| 近因 | `recent(n=5) -> List[Episode]` | 最近 n 条，新到旧 |
| 回放 | `sample(n=32) -> List[Episode]` | 均匀随机（seed 可复现） |
| 存取 | `state_dict() / load_state_dict(sd)` | 纯 NumPy + 元数据，可 pickle |
| 其他 | `clear()` / `__len__` | 清空（时间钟不回拨）/ 当前行数 |

`Episode`：`z`（事件向量）、`t`（写入序号）、`meta`（任意键值）、
`get(key, default)`。

`Retrieved`：`episodes`、`sims`（对应相似度）、`best`（最相似或
None）、`__bool__`（空检索为假）。

行为细节：容量满按 FIFO 淘汰最旧；零向量查询相似度为 0；
空记忆 `read` 返回空、`sample` 抛错。

异常：`MemoryError(ValueError)` —— 非法容量/维度、z 或 query
形状不匹配、NaN/Inf、非正 k/n。
