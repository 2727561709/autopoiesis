"""M04 接口契约测试。"""
import inspect

import numpy as np

from cog_perception import Encoder


def test_api_surface():
    for name in ("forward", "__call__", "backward", "parameters",
                 "zero_grad", "sgd_step", "state_dict", "load_state_dict"):
        assert hasattr(Encoder, name)


def test_forward_signature():
    sig = inspect.signature(Encoder.forward)
    assert list(sig.parameters)[:2] == ["self", "obs"]


def test_obs_z_contract():
    enc = Encoder(obs_dim=6, latent_dim=16, hidden_dim=8, seed=0)
    z = enc(np.ones(6))
    assert isinstance(z, np.ndarray)
    assert z.shape == (enc.latent_dim,)
