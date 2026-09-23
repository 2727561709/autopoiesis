"""cog-drives: M12 驱动系统模块公开 API。
cog-drives: public API of the M12 drive-system module.

用法 / usage:
    from cog_drives import DriveSystem
    drives = DriveSystem(decay=0.995)
    drives.step({"energy": +0.3})       # 吃东西 / eat something
    gap = drives.deficit()               # 缺口 / deficits
    vec = drives.to_tensor()             # (5,)
"""
from .core import DriveSystem
from .types import DRIVE_NAMES, DriveError, Effects

__all__ = ["DriveSystem", "DRIVE_NAMES", "Effects", "DriveError"]
__version__ = "0.1.0"
