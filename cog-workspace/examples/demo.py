"""M14 最小可运行示例：python examples/demo.py"""
import numpy as np

from cog_workspace import Workspace

if __name__ == "__main__":
    ws = Workspace(
        source_dims={"perc": 16, "mem": 16, "emo": 4, "goal": 8},
        broadcast_dim=32, seed=0,
    )
    rng = np.random.default_rng(1)

    # 平静状态：普通感知 + 记忆 + 目标
    b, w = ws(perc=rng.normal(size=16), mem=rng.normal(size=16),
              emo=np.zeros(4), goal=np.ones(8))
    print("平静:", {k: round(v, 3) for k, v in w.items()})

    # 强情绪状态：情绪源幅度大 => 竞争中胜出
    b, w = ws(perc=rng.normal(size=16), mem=rng.normal(size=16),
              emo=np.full(4, 10.0), goal=np.ones(8))
    print("强情绪:", {k: round(v, 3) for k, v in w.items()})

    # 只有感知参与
    b, w = ws(perc=np.ones(16))
    print("仅感知:", w, "broadcast:", b.round(2)[:6], "...")
