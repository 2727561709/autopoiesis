"""cog-perception: M04 感知编码模块公开 API。
cog-perception: public API of the M04 perception module.

用法 / usage:
    from cog_perception import Encoder
    enc = Encoder(obs_dim=16, latent_dim=128)
    z = enc(obs)            # (128,)  值域 (-1, 1) / in (-1, 1)
"""
from .core import Encoder
from .types import Observation, PerceptionError

__all__ = ["Encoder", "Observation", "PerceptionError"]
__version__ = "0.1.0"
