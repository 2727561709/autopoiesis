"""M10 接口契约测试。"""
import inspect

import numpy as np

from cog_world_model import ImagineResult, WorldModel, WorldModelError


def test_api_surface():
    for name in ("forward", "__call__", "observe", "train_step",
                 "prediction_error", "imagine", "parameters", "zero_grad",
                 "state_dict", "load_state_dict"):
        assert hasattr(WorldModel, name)


def test_forward_signature():
    sig = inspect.signature(WorldModel.forward)
    assert list(sig.parameters)[:3] == ["self", "z", "a"]


def test_imagine_signature():
    sig = inspect.signature(WorldModel.imagine)
    assert list(sig.parameters) == ["self", "z", "policy", "horizon"]


def test_result_contract():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    res = wm.imagine(np.zeros(4), policy=lambda z: 1, horizon=5)
    assert isinstance(res, ImagineResult)
    assert len(res.zs) == 6            # H + 1（含初始 z）
    assert len(res.actions) == 5
    assert isinstance(res.total_reward, float)
    assert isinstance(res.terminated, bool)


def test_error_type():
    assert issubclass(WorldModelError, ValueError)
