"""M10 核心行为测试：转移学习 / 奖励学习 / 想象 rollout。"""
import numpy as np

from cog_world_model import WorldModel


def _simple_env(rng, z, a_id):
    """确定性玩具环境：z' = 0.5 * roll(z)；动作 0 加食物回报。
    Deterministic toy env: z' = 0.5 * roll(z); action 0 yields a food
    reward."""
    z_next = 0.5 * np.roll(z, 1)
    r = 0.8 if a_id == 0 else -0.2
    return z_next, r, 0.0


def test_forward_shapes_and_bounds():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    z = np.array([0.1, -0.2, 0.3, -0.4])
    zp, r, d = wm(z, 2)                       # int 动作 one-hot
    assert zp.shape == (4,)
    assert np.all(np.abs(zp) < 1.0)           # tanh 有界
    assert isinstance(r, float) and isinstance(d, float)
    assert 0.0 < d < 1.0                      # (tanh+1)/2
    # 批量 / batch
    zb = np.stack([z, -z])
    zpb, rb, db = wm(zb, [0, 1])              # 批量离散动作
    assert zpb.shape == (2, 4) and rb.shape == (2,) and db.shape == (2,)


def _sample_err(wm, rng, n=20):
    """在玩具环境上测量平均预测误差。Measure mean prediction error on
    the toy env."""
    errs = []
    for _ in range(n):
        z = rng.uniform(-0.8, 0.8, size=4)
        a_id = int(rng.integers(0, 2))
        z_next, _, _ = _simple_env(rng, z, a_id)
        errs.append(wm.prediction_error(z, a_id, z_next))
    return float(np.mean(errs))


def test_learns_linear_transition():
    wm = WorldModel(latent_dim=4, action_dim=2, hidden_dim=32, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        z = rng.uniform(-0.8, 0.8, size=4)
        a_id = int(rng.integers(0, 2))
        z_next, r, _ = _simple_env(rng, z, a_id)
        wm.observe(z, a_id, z_next, reward=r)
    err0 = _sample_err(wm, rng)
    for _ in range(400):
        wm.train_step(lr=0.05, batch_size=32)
    err1 = _sample_err(wm, rng)
    assert err1 < err0                         # 预测误差下降
    assert err1 < 0.01                        # 足够准确


def test_learns_reward():
    wm = WorldModel(latent_dim=4, action_dim=2, hidden_dim=32, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(200):
        z = rng.uniform(-0.8, 0.8, size=4)
        a_id = int(rng.integers(0, 2))
        z_next, r, _ = _simple_env(rng, z, a_id)
        wm.observe(z, a_id, z_next, reward=r)
    for _ in range(300):
        wm.train_step(lr=0.05, batch_size=32)
    z = np.zeros(4)
    _, r0, _ = wm(z, 0)
    _, r1, _ = wm(z, 1)
    assert r0 > 0.3 > r1                      # 学到"动作 0 有回报"
    assert abs(r0 - 0.8) < 0.25 and abs(r1 - (-0.2)) < 0.25


def test_imagine_rollout_and_termination():
    wm = WorldModel(latent_dim=4, action_dim=2, hidden_dim=8, seed=0)
    z = np.array([0.5, -0.5, 0.2, -0.2])
    res = wm.imagine(z, policy=lambda zz: 0, horizon=10)
    assert len(res.zs) == 11 and len(res.actions) == 10
    assert not res.terminated
    # 推高 done 头输出 => rollout 提前终止
    wm.done_net.layers[2].b += 5.0
    res2 = wm.imagine(z, policy=lambda zz: 0, horizon=10)
    assert res2.terminated and len(res2.actions) == 1


def test_state_dict_roundtrip():
    wm = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=0)
    z = np.array([0.1, 0.2, -0.1, -0.2])
    wm.observe(z, 1, 0.5 * np.roll(z, 1), reward=0.5)
    sd = wm.state_dict()
    wm2 = WorldModel(latent_dim=4, action_dim=3, hidden_dim=8, seed=99)
    wm2.load_state_dict(sd)
    assert np.allclose(wm(z, 1)[0], wm2(z, 1)[0])
    assert len(wm2._buf) == 1                 # 缓冲一并保存


def test_train_step_returns_losses():
    wm = WorldModel(latent_dim=4, action_dim=2, hidden_dim=8, seed=0)
    rng = np.random.default_rng(3)
    for _ in range(10):
        z = rng.uniform(-0.5, 0.5, size=4)
        wm.observe(z, 0, 0.5 * z, reward=0.1)
    loss = wm.train_step(lr=0.01)
    assert set(loss) == {"trans", "reward", "done"}
    assert all(v >= 0.0 for v in loss.values())
