"""cog-workspace: M14 全局工作空间模块公开 API。
cog-workspace: public API of the M14 global-workspace module.

用法 / usage:
    from cog_workspace import Workspace
    ws = Workspace(source_dims={"perc": 128, "mem": 128, "emo": 8, "goal": 32})
    broadcast, weights = ws(perc=z, mem=z_mem, emo=e, goal=g)
    # weights 例 / e.g.: {"perc": 0.42, "mem": 0.31, "emo": 0.05, "goal": 0.22}
"""
from .core import Workspace
from .types import SOURCE_NAMES, BroadcastResult, SourceVector, WorkspaceError

__all__ = ["Workspace", "SOURCE_NAMES", "SourceVector", "BroadcastResult", "WorkspaceError"]
__version__ = "0.1.0"
