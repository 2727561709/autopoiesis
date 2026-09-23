"""homeo_rl —— 阶段 A：内稳态 RL + 预测误差辅助。

验证"新物种"训练范式：模型的唯一回报来自内稳态缺口的缩减，
没有任何人工任务奖励；预测误差提供好奇式的内在驱动。
"""
from .drives import DEFAULT_SETPOINTS, deficits, homeostatic_reward, total_deficit
from .env import ACTION_NAMES, N_ACTIONS, EnvConfig, HomeostaticGridWorld
from .model import HomeoActorCritic

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
