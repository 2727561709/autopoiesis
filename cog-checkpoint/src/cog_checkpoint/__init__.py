"""cog-checkpoint: M03 存档模块公开 API。

用法:
    from cog_checkpoint import Checkpoint
    Checkpoint.save({"step": 10}, "ckpt/agent.ckpt")
    state = Checkpoint.load("ckpt/agent.ckpt")
"""
from .core import Checkpoint
from .types import FORMAT_VERSION, CheckpointError, CheckpointMeta

__all__ = ["Checkpoint", "CheckpointError", "CheckpointMeta", "FORMAT_VERSION"]
__version__ = "0.1.0"
