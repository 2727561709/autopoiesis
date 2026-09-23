"""M05 边界条件测试。"""
import numpy as np
import pytest

from cog_policy import DiagGaussian, Policy, PolicyError


def test_wrong_latent_dim_raises():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    with pytest.raises(PolicyError, match="expected last dim"):
        pol(np.zeros(9))


def test_nan_z_raises():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    with pytest.raises(PolicyError, match="NaN"):
        pol(np.array([np.nan] * 8))


def test_invalid_constructor_dims():
    with pytest.raises(PolicyError):
        Policy(latent_dim=0, action_dim=4)
    with pytest.raises(PolicyError):
        Policy(latent_dim=8, action_dim=-1)


def test_load_mismatch_raises():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    other = Policy(latent_dim=8, action_dim=6, hidden_dim=8, seed=0)
    with pytest.raises(PolicyError, match="mismatch"):
        other.load_state_dict(pol.state_dict())


def test_mean_bounded_by_tanh():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    for scale in (1.0, 10.0, 100.0):
        dist, _ = pol(np.full(8, scale))
        assert np.all(np.abs(dist.mean) <= 1.0)


def test_log_prob_batch_vs_single_consistency():
    d = DiagGaussian(mean=np.zeros((3, 2)), std=np.ones((3, 2)))
    actions = np.arange(6, dtype=float).reshape(3, 2)
    batch_lp = d.log_prob(actions)
    for i in range(3):
        single = DiagGaussian(mean=d.mean[i], std=d.std[i])
        assert single.log_prob(actions[i]) == pytest.approx(batch_lp[i])


def test_extreme_std_log_prob_finite():
    d = DiagGaussian(mean=np.zeros(2), std=np.array([1e-3, 1e3]))
    lp = d.log_prob(np.zeros(2))
    assert np.isfinite(lp)


def test_extreme_latent_values():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    dist, value = pol(np.full(8, 1e6))
    assert np.all(np.isfinite(dist.mean))
    assert np.isfinite(value)
