"""M05 接口契约测试。"""
import inspect

import numpy as np

from cog_policy import DiagGaussian, Policy


def test_api_surface():
    for name in ("forward", "__call__", "act", "backward", "parameters",
                 "zero_grad", "sgd_step", "state_dict", "load_state_dict"):
        assert hasattr(Policy, name)


def test_forward_signature():
    sig = inspect.signature(Policy.forward)
    assert list(sig.parameters)[:2] == ["self", "z"]


def test_dist_api():
    for name in ("sample", "log_prob", "entropy", "deterministic"):
        assert hasattr(DiagGaussian, name)


def test_policy_z_to_dist_value_contract():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    dist, value = pol(np.zeros(8))
    assert dist.mean.shape == (4,)
    assert np.ndim(value) == 0
