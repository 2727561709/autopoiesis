"""homeo_rl —— 阶段 A：内稳态 RL + 预测误差辅助。
homeo_rl — Stage A: homeostatic RL + prediction-error assistance.

验证"新物种"训练范式：模型的唯一回报来自内稳态缺口的缩减，
没有任何人工任务奖励；预测误差提供好奇式的内在驱动。
Validates the "new-species" training paradigm: the model's only reward
comes from homeostatic deficit reduction — no artificial task rewards
whatsoever; prediction error provides a curiosity-style intrinsic drive.

注意：模型（torch）按需惰性导入，环境与对比工具无需安装 torch。
Note: the model (torch) is imported lazily on demand, so the env and
comparison tools work without torch installed.
"""
from .drives import DEFAULT_SETPOINTS, deficits, homeostatic_reward, total_deficit
from .env import ACTION_NAMES, N_ACTIONS, EnvConfig, HomeostaticGridWorld

__all__ = [
    "ACTION_NAMES",
    "N_ACTIONS",
    "DEFAULT_SETPOINTS",
    "EnvConfig",
    "HomeostaticGridWorld",
    "HomeoActorCritic",
    "deficits",
    "homeostatic_reward",
    "total_deficit",
]


def __getattr__(name: str):
    """惰性导入 HomeoActorCritic（避免强制依赖 torch）。
    Lazily import HomeoActorCritic (torch is not a hard dependency)."""
    if name == "HomeoActorCritic":
        from .model import HomeoActorCritic
        return HomeoActorCritic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
