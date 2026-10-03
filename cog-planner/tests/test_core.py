"""M16 核心行为测试：在 stub 世界模型上做想象规划。"""
import numpy as np

from cog_planner import Planner

from stub_wm import StubWM


def test_planner_finds_high_reward_action():
    pl = Planner(StubWM(), action_dim=2, horizon=5, population=32,
                 generations=5, seed=0)
    res = pl.plan(np.array([0.2, -0.3, 0.1, -0.1]))
    # 玩具世界里动作 0 每步 +0.8 => 全 0 是最优
    # In the toy world action 0 gives +0.8/step: all-zeros is optimal
    ids = res.actions.argmax(axis=-1)
    assert (ids == 0).mean() > 0.8
    assert res.score > 5 * 0.5                 # 接近 5 * 0.8


def test_planner_beats_random_baseline():
    wm = StubWM()
    pl = Planner(wm, action_dim=2, horizon=6, population=32,
                 generations=5, seed=1)
    z = np.array([0.2, -0.3, 0.1, -0.1])
    res = pl.plan(z)
    rng = np.random.default_rng(9)
    n_win = 0
    for _ in range(20):                        # 规划应优于随机序列
        rand_score = 0.0
        cur = z
        for t in range(6):
            zp, r, _ = wm(cur, np.eye(2)[rng.integers(0, 2)])
            rand_score += r
            cur = zp
        if res.score > rand_score:
            n_win += 1
    assert n_win >= 18                          # 至少 90% 胜出


def test_goal_directed_planning():
    """goal 给定时，终局应比纯随机序列更接近目标。
    With a goal, the plan's endpoint should beat random sequences."""
    wm = StubWM()
    z = np.array([0.5, -0.5, 0.2, -0.2])
    goal = -z                                   # 目标 = 翻转初始状态
    pl = Planner(wm, action_dim=2, horizon=4, population=32,
                 generations=6, goal_weight=4.0, seed=2)
    res = pl.plan(z, goal=goal)
    d_plan = float(np.linalg.norm(res.zs[-1] - goal))
    rng = np.random.default_rng(7)
    d_rand = []
    for _ in range(20):
        cur = z
        for t in range(4):
            cur = wm(cur, np.eye(2)[rng.integers(0, 2)])[0]
        d_rand.append(float(np.linalg.norm(cur - goal)))
    assert d_plan <= np.mean(d_rand)            # 不输随机均值


def test_continuous_mode():
    pl = Planner(StubWM(action_dim=3), action_dim=3, discrete=False,
                 horizon=4, population=32, generations=4, seed=0)
    res = pl.plan(np.zeros(4))
    assert res.actions.shape == (4, 3)
    assert np.all(np.abs(res.actions) <= 1.0)   # 裁剪到 [-1, 1]
    assert np.isfinite(res.score)


def test_history_and_first_action():
    pl = Planner(StubWM(), action_dim=2, horizon=5, population=24,
                 generations=4, seed=0)
    res = pl.plan(np.zeros(4))
    assert len(res.history) == 4
    assert res.score >= res.history[0] - 1e-9   # 最优不低于首代
    assert res.first_action.shape == (2,)
