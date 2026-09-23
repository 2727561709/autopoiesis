"""M04 边界条件测试。"""
import numpy as np
import pytest

from cog_perception import Encoder, PerceptionError


def test_wrong_dim_raises():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=0)
    with pytest.raises(PerceptionError, match="expected last dim"):
        enc(np.zeros(5))


def test_nan_inf_raises():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=0)
    with pytest.raises(PerceptionError, match="NaN"):
        enc(np.array([np.nan, 0.0, 0.0, 0.0]))
    with pytest.raises(PerceptionError, match="NaN"):
        enc(np.array([np.inf, 0.0, 0.0, 0.0]))


def test_3d_input_raises():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=0)
    with pytest.raises(PerceptionError, match="1-D or 2-D"):
        enc(np.zeros((2, 2, 4)))


def test_invalid_dims_at_construction():
    with pytest.raises(PerceptionError):
        Encoder(obs_dim=0, latent_dim=8)
    with pytest.raises(PerceptionError):
        Encoder(obs_dim=4, latent_dim=-1)


def test_multimodal_dict_obs():
    enc = Encoder(obs_dim=7, latent_dim=8, hidden_dim=8, seed=0)
    obs = {"vision": np.array([1.0, 2.0]), "audio": np.array([3.0]),
           "proprio": np.array([4.0, 5.0, 6.0, 7.0])}
    z = enc(obs)
    assert z.shape == (8,)
    # 字典拼接应与手工拼接一致（按插入顺序）
    flat = np.concatenate([obs["vision"], obs["audio"], obs["proprio"]])
    assert np.allclose(z, enc(flat))


def test_empty_dict_obs_raises():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=0)
    with pytest.raises(PerceptionError, match="empty"):
        enc({})


def test_extreme_input_values():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=0)
    z = enc(np.array([1e8, -1e8, 1e-8, 0.0]))
    assert np.all(np.isfinite(z))


def test_list_input_accepted():
    enc = Encoder(obs_dim=3, latent_dim=4, hidden_dim=4, seed=0)
    z = enc([1.0, 2.0, 3.0])
    assert z.shape == (4,)


def test_backward_before_forward_raises():
    enc = Encoder(obs_dim=3, latent_dim=4, hidden_dim=4, seed=0)
    with pytest.raises(RuntimeError, match="before forward"):
        enc.backward(np.zeros(4))
