"""cog-working-memory: M06 工作记忆模块公开 API。

用法:
    from cog_working_memory import WorkingMemory
    wm = WorkingMemory(capacity=16, dim=128)
    wm.write(z)                  # 写入（满则 FIFO 驱逐）
    z_out = wm.read(query)       # 注意力加权聚合读出
    w = wm.attention(query)      # 注意力权重（和为 1）
"""
from .core import TEMPERATURE, WorkingMemory
from .types import Vector, WorkingMemoryError

__all__ = ["WorkingMemory", "Vector", "WorkingMemoryError", "TEMPERATURE"]
__version__ = "0.1.0"
