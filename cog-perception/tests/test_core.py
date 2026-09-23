"""M04 核心功能测试：编码一致性、维度、梯度。"""
import numpy as np
import pytest

from cog_perception import Encoder, PerceptionError


def test_output_shape_single_and_batch():
    enc = Encoder(obs_dim=8, latent_dim=32, hidden_dim=16, seed=1)
    z = enc(np.zeros(8))
    assert z.shape == (32,)
    zb = enc(np.zeros((5, 8)))
    assert zb.shape == (5, 32)


def test_bounded_output():
    enc = Encoder(obs_dim=8, latent_dim=32, hidden_dim=16, seed=1)
    z = enc(np.random.default_rng(0).normal(size=(64, 8)))
    assert np.all(z > -1.0) and np.all(z < 1.0)


def test_encoding_consistency_same_seed():
    a = Encoder(obs_dim=8, latent_dim=16, hidden_dim=8, seed=42)
    b = Encoder(obs_dim=8, latent_dim=16, hidden_dim=8, seed=42)
    obs = np.arange(8, dtype=float)
    assert np.allclose(a(obs), b(obs))


def test_different_seed_different_encoding():
    a = Encoder(obs_dim=8, latent_dim=16, hidden_dim=8, seed=1)
    b = Encoder(obs_dim=8, latent_dim=16, hidden_dim=8, seed=2)
    obs = np.arange(8, dtype=float)
    assert not np.allclose(a(obs), b(obs))


def test_deterministic_forward():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=3)
    obs = np.array([1.0, -2.0, 0.5, 0.0])
    assert np.allclose(enc(obs), enc(obs))


def test_batch_equals_single():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=3)
    obs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    zb = enc(obs)
    for i in range(2):
        assert np.allclose(enc(obs[i]), zb[i])


def test_gradients_analytic_vs_numeric():
    enc = Encoder(obs_dim=3, latent_dim=2, hidden_dim=4, seed=7)
    obs = np.array([0.3, -0.7, 0.1])
    target = np.array([0.5, -0.5])
    # loss = 0.5 * ||z - target||^2
    def loss():
        z = enc(obs)
        return 0.5 * float(((z - target) ** 2).sum())

    base = loss()
    enc.zero_grad()
    z = enc(obs)
    enc.backward(z - target)  # dL/dz

    W1 = enc.net.layers[0].W
    i, j = 1, 2
    eps = 1e-6
    old = W1[i, j]
    W1[i, j] = old + eps; lp = loss()
    W1[i, j] = old - eps; lm = loss()
    W1[i, j] = old
    numeric = (lp - lm) / (2 * eps)
    analytic = enc.net.layers[0].dW[i, j]
    assert analytic == pytest.approx(numeric, abs=1e-5)


def test_sgd_reduces_loss():
    enc = Encoder(obs_dim=3, latent_dim=2, hidden_dim=8, seed=7)
    obs = np.array([0.3, -0.7, 0.1])
    target = np.array([0.5, -0.5])

    def loss():
        z = enc(obs)
        return 0.5 * float(((z - target) ** 2).sum())

    before = loss()
    for _ in range(200):
        enc.zero_grad()
        z = enc(obs)
        enc.backward(z - target)
        enc.sgd_step(lr=0.1)
    assert loss() < before


def test_state_dict_roundtrip():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=9)
    sd = enc.state_dict()
    enc2 = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=100)
    enc2.load_state_dict(sd)
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(enc(obs), enc2(obs))


def test_load_mismatch_raises():
    enc = Encoder(obs_dim=4, latent_dim=8, hidden_dim=8, seed=9)
    enc2 = Encoder(obs_dim=5, latent_dim=8, hidden_dim=8, seed=9)
    with pytest.raises(PerceptionError, match="mismatch"):
        enc2.load_state_dict(enc.state_dict())
