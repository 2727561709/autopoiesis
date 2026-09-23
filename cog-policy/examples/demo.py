"""M05 最小可运行示例：python examples/demo.py"""
import numpy as np

from cog_policy import Policy

if __name__ == "__main__":
    pol = Policy(latent_dim=16, action_dim=4, hidden_dim=32, seed=0)
    rng = np.random.default_rng(0)

    z = np.random.default_rng(1).normal(size=16)
    dist, value = pol(z)
    print("mean:", dist.mean.round(3))
    print("std :", dist.std.round(3))
    print("value:", round(float(value), 4))

    # 随机采样 5 步
    for t in range(5):
        a, lp, v = pol.act(z, rng)
        print(f"t={t} action={a.round(3)} log_prob={lp:.3f} value={v:.3f}")

    # 确定性动作
    a, lp, v = pol.act(z, rng, deterministic=True)
    print("deterministic action:", a.round(3), "log_prob:", round(lp, 3))
