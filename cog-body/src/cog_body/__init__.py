"""cog-body: M25 虚拟身体模块公开 API。
cog-body: public API of the M25 virtual-body module.

用法 / usage:
    from cog_body import Body
    body = Body(position=(1, 1), arena_size=10)
    effects = body.step({"move": (0.5, 0.0)})   # -> {"energy": -0.01, "integrity": 0.0}
"""
from .core import MOVE_COST_PER_UNIT, REST_RECOVERY, Body
from .types import Action, BodyError, Effects, Position

__all__ = ["Body", "Action", "Effects", "Position", "BodyError",
           "MOVE_COST_PER_UNIT", "REST_RECOVERY"]
__version__ = "0.1.0"
