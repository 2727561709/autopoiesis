"""cog-policy: M05 行动策略模块公开 API。

用法:
    from cog_policy import Policy
    pol = Policy(latent_dim=128, action_dim=4)
    dist, value = pol(z)
    action, log_prob, value = pol.act(z, rng)
"""
from .core import LOG_STD_MAX, LOG_STD_MIN, Policy
from .types import DiagGaussian, PolicyError

__all__ = ["Policy", "DiagGaussian", "PolicyError", "LOG_STD_MIN", "LOG_STD_MAX"]
__version__ = "0.1.0"
