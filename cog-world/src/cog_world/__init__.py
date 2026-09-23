"""cog-world: M26 世界环境模块公开 API。

用法:
    from cog_world import World
    world = World(size=10, n_food=10, seed=0)
    aid = world.add_agent()
    obs = world.observe(aid)                  # (6,)
    obs, reward = world.step(aid, {"move": (0.5, 0.0)})
"""
from .core import OBS_DIM, DANGER_DAMAGE, DANGER_RADIUS, EAT_RADIUS, World
from .types import Action, StepResult, WorldError

__all__ = ["World", "OBS_DIM", "Action", "StepResult", "WorldError",
           "EAT_RADIUS", "DANGER_RADIUS", "DANGER_DAMAGE"]
__version__ = "0.1.0"
