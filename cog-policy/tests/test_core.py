"""M05 核心功能测试：采样、log_prob、价值形状。"""
import numpy as np
import pytest

from cog_policy import DiagGaussian, Policy, PolicyError


def test_forward_returns_dist_and_value():
    pol = Policy(latent_dim=16, action_dim=3, hidden_dim=8, seed=0)
    z = np.random.default_rng(1).normal(size=16)
    dist, value = pol(z)
    assert isinstance(dist, DiagGaussian)
    assert dist.mean.shape == (3,)
    assert np.ndim(value) == 0  # 标量


def test_sample_shape_and_finiteness():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    rng = np.random.default_rng(2)
    z = np.zeros(8)
    dist, _ = pol(z)
    a = dist.sample(rng)
    assert a.shape == (4,)
    assert np.all(np.isfinite(a))


def test_log_prob_matches_formula():
    mean = np.array([0.0, 1.0])
    std = np.array([0.5, 2.0])
    d = DiagGaussian(mean=mean, std=std)
    a = np.array([0.5, -1.0])
    lp = d.log_prob(a)
    expected = (-0.5 * ((0.5 / 0.5) ** 2 + np.log(2 * np.pi * 0.25))
                - 0.5 * ((-2.0 / 2.0) ** 2 + np.log(2 * np.pi * 4.0)))
    assert lp == pytest.approx(expected)


def test_log_prob_max_at_mean():
    d = DiagGaussian(mean=np.zeros(3), std=np.ones(3))
    assert d.log_prob(np.zeros(3)) > d.log_prob(np.ones(3))


def test_batch_forward():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    z = np.random.default_rng(3).normal(size=(10, 8))
    dist, value = pol(z)
    assert dist.mean.shape == (10, 4)
    assert value.shape == (10,)
    lp = dist.log_prob(np.zeros((10, 4)))
    assert lp.shape == (10,)


def test_sampling_reproducible_with_seed():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    z = np.ones(8)
    a1, lp1, v1 = pol.act(z, np.random.default_rng(42))
    a2, lp2, v2 = pol.act(z, np.random.default_rng(42))
    assert np.allclose(a1, a2) and lp1 == lp2 and v1 == v2


def test_deterministic_act():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    z = np.ones(8)
    a, lp, v = pol.act(z, np.random.default_rng(0), deterministic=True)
    dist, _ = pol(z)
    assert np.allclose(a, dist.deterministic())
    assert np.allclose(a, dist.mean)  # tanh 输出即 mean


def test_entropy_positive():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    dist, _ = pol(np.zeros(8))
    assert dist.entropy() > 0


def test_std_always_positive():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    pol.log_std = np.array([-100.0, 100.0, 0.0, 0.0])  # 极端值被限幅
    dist, _ = pol(np.zeros(8))
    assert np.all(dist.std > 0)
    assert np.all(np.isfinite(dist.std))


def test_state_dict_roundtrip():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=5)
    pol2 = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=99)
    pol2.load_state_dict(pol.state_dict())
    z = np.random.default_rng(1).normal(size=8)
    d1, v1 = pol(z)
    d2, v2 = pol2(z)
    assert np.allclose(d1.mean, d2.mean) and v1 == pytest.approx(v2)


def test_sgd_step_runs_and_changes_params():
    pol = Policy(latent_dim=8, action_dim=4, hidden_dim=8, seed=0)
    p0 = pol.mu_net.layers[0].W.copy()
    pol.zero_grad()
    pol.forward(np.ones(8))
    pol.backward(np.ones(4), np.ones(1), np.ones(4))
    pol.sgd_step(lr=0.1)
    assert not np.allclose(p0, pol.mu_net.layers[0].W)
