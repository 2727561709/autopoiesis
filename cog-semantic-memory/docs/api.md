# cog-semantic-memory API

## `SemanticMemory(decay=0.5, threshold=0.05)`

| 方法 | 签名 | 说明 |
|---|---|---|
| 概念 | `add_concept(cid, **attrs) -> Concept` | 幂等：存在则合并属性 |
| 边 | `link(src, dst, relation="rel", weight=1.0, bidirectional=False)` | 概念须已存在，禁自环 |
| 查询 | `get(cid) / neighbors(cid) / in / len` | 取概念 / 出边邻居 / 包含判断 |
| 检索 | `spread(seeds, depth=3) -> {cid: activation}` | 扩散激活，按激活降序 |

`seeds`：`{cid: 初始激活}` 字典或 `List[cid]`（各取 1.0）。

传播规则：每跳 `activation × weight × decay` 累加到邻居；激活只增
不减；`< threshold` 的概念不进入结果；结果按激活值降序。

异常：`SemanticError(ValueError)` —— decay/threshold/weight 越界、
空或非法 cid、未知概念/种子、自环、负或 NaN 激活、负 depth。
