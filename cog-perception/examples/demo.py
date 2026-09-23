"""M04 最小可运行示例：python examples/demo.py"""
import numpy as np

from cog_perception import Encoder

if __name__ == "__main__":
    enc = Encoder(obs_dim=10, latent_dim=128, hidden_dim=64, seed=0)

    # 单个向量观测
    obs = np.random.default_rng(1).normal(size=10)
    z = enc(obs)
    print("single obs  -> z:", z.shape, "range:", (z.min(), z.max()))

    # 批量
    zb = enc(np.random.default_rng(2).normal(size=(8, 10)))
    print("batch (8,10) -> z:", zb.shape)

    # 多模态字典
    zm = enc({"vision": obs[:6], "audio": obs[6:]})
    print("multimodal  -> z:", zm.shape)

    # 简单训练：让 z 靠近目标
    target = np.zeros(128)
    for i in range(300):
        enc.zero_grad()
        zz = enc(obs)
        enc.backward(zz - target)
        enc.sgd_step(lr=0.05)
    print("train loss:", float(((enc(obs) - target) ** 2).mean()))
