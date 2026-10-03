"""M10 边界与防御测试。"""
import numpy as np
import pytest

from cog_world_model import WorldModel, WorldModelError


def test_bad_constructor():
    with pytest.raises(WorldModelError):
        WorldModel(latent_dim=0, action_dim=3)
    with pytest.raises(WorldModelError):
        WorldModel(latent_dim=3, action_dim=-1)


def test_z_dim_mismatch():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm(np.ones(5), 0)


def test_action_out_of_range():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm(np.ones(4), 3)                       # 越界离散动作
    with pytest.raises(WorldModelError):
        wm(np.ones(4), -1)
    with pytest.raises(WorldModelError):
        wm(np.ones(4), np.ones(5))              # 连续动作维度错


def test_nan_inputs():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    z = np.ones(4); z[0] = np.nan
    with pytest.raises(WorldModelError):
        wm(z, 0)
    with pytest.raises(WorldModelError):
        wm.observe(z, 0, np.zeros(4))


def test_single_batch_mix():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm(np.ones(4), np.ones((2, 3)))         # 单 z + 批 a
    with pytest.raises(WorldModelError):
        wm(np.ones((2, 4)), 0)                 # 批 z + 单 a


def test_batch_size_mismatch():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm(np.ones((2, 4)), np.ones((3, 3)))


def test_empty_buffer_train():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm.train_step()


def test_observe_shape_mismatch():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm.observe(np.ones(4), 0, np.ones(5))
    with pytest.raises(WorldModelError):
        wm.prediction_error(np.ones(4), 0, np.ones(5))


def test_bad_horizon():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm.imagine(np.ones(4), policy=lambda z: 0, horizon=-1)


def test_zero_horizon():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    res = wm.imagine(np.ones(4), policy=lambda z: 0, horizon=0)
    assert len(res.zs) == 1 and not res.actions and not res.terminated


def test_state_dict_mismatch():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    sd = wm.state_dict()
    wm2 = WorldModel(latent_dim=5, action_dim=3, hidden_dim=8, seed=0)
    with pytest.raises(WorldModelError):
        wm2.load_state_dict(sd)
