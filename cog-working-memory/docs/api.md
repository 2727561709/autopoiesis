# cog-working-memory API

## `WorkingMemory(capacity=16, dim=None)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 写入 | `write(z) -> None` | 满则驱逐最旧（FIFO） |
| 读取 | `read(query) -> (dim,)` | `Σ softmax(cos(q, sᵢ)) · sᵢ` |
| 注意力 | `attention(query) -> (n,)` | 权重和为 1 |
| 维护 | `clear()` / `__len__` / `state()` | state: size/capacity/dim |

异常：`WorkingMemoryError(ValueError)` —— 空读、维度不匹配、NaN/Inf、非法构造参数。
