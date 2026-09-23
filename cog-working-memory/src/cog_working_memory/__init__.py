"""cog-working-memory: M06 工作记忆模块公开 API。
cog-working-memory: public API of the M06 working-memory module.

用法 / usage:
    from cog_working_memory import WorkingMemory
    wm = WorkingMemory(capacity=16, dim=128)
    wm.write(z)                  # 写入（满则 FIFO 驱逐）/ write (FIFO eviction when full)
    z_out = wm.read(query)       # 注意力加权聚合读出 / attention-weighted read
    w = wm.attention(query)      # 注意力权重（和为 1）/ attention weights (sum to 1)
"""
from .core import TEMPERATURE, WorkingMemory
from .types import Vector, WorkingMemoryError

__all__ = ["WorkingMemory", "Vector", "WorkingMemoryError", "TEMPERATURE"]
__version__ = "0.1.0"
