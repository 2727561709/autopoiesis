"""cog-perception: M04 感知编码模块公开 API。

用法:
    from cog_perception import Encoder
    enc = Encoder(obs_dim=16, latent_dim=128)
    z = enc(obs)            # (128,)  值域 (-1, 1)
"""
from .core import Encoder
from .types import Observation, PerceptionError

__all__ = ["Encoder", "Observation", "PerceptionError"]
__version__ = "0.1.0"
