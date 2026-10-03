"""M10 最小可运行示例：python examples/demo.py"""
import numpy as np

from cog_world_model import WorldModel

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wm = WorldModel(latent_dim=4, action_dim=2, hidden_dim=32, seed=0)

    # 玩具环境：z' = 0.5 * roll(z)，动作 0 有回报
    def env(z, a):
        return 0.5 * np.roll(z, 1), (0.8 if a == 0 else -0.2), 0.0

    # 1) 记录真实转移
    for _ in range(200):
        z = rng.uniform(-0.8, 0.8, size=4)
        a = int(rng.integers(0, 2))
        z_next, r, d = env(z, a)
        wm.observe(z, a, z_next, reward=r, done=d)

    # 2) 训练前后的预测误差（"好奇"接口）
    z = rng.uniform(-0.8, 0.8, size=4)
    z_next, r, _ = env(z, 0)
    print("err before:", round(wm.prediction_error(z, 0, z_next), 4))
    for _ in range(400):
        loss = wm.train_step(lr=0.05, batch_size=32)
    print("err after :", round(wm.prediction_error(z, 0, z_next), 4))
    print("last loss:", {k: round(v, 4) for k, v in loss.items()})

    # 3) 想象 rollout：贪心选预测回报更高的动作
    res = wm.imagine(
        z, policy=lambda zz: 0 if wm(zz, 0)[1] > wm(zz, 1)[1] else 1,
        horizon=8)
    print("imagine: steps =", len(res.actions),
          " total_reward =", round(res.total_reward, 3),
          " terminated =", res.terminated)
