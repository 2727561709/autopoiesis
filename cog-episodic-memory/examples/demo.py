"""M07 最小可运行示例：python examples/demo.py

模拟智能体在四个"地点"的经历，演示检索 / 近因 / 回放。
Simulate experiences at four "places"; demo retrieval, recency,
and replay.
"""
import numpy as np

from cog_episodic import EpisodicMemory

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mem = EpisodicMemory(capacity=1000, latent_dim=16, seed=0)

    places = {"food": np.ones(16), "water": -np.ones(16),
              "nest": np.eye(16)[0], "hazard": np.eye(16)[5]}
    for i in range(40):
        name = ["food", "water", "nest", "hazard"][i % 4]
        z = places[name] + 0.05 * rng.standard_normal(16)
        mem.write(z, place=name, reward=1.0 if name == "food" else 0.0,
                  episode=i)

    # 1) 相似检索：饿了的智能体想起食物在哪
    query = places["food"] + 0.1 * rng.standard_normal(16)
    res = mem.read(query, k=3)
    print("query ~ food, top-3:", [(e.get("place"), round(s, 3))
                                    for e, s in zip(res.episodes, res.sims)])

    # 2) 近因：最近 3 段经历
    print("recent:", [(e.get("place"), e.get("episode"))
                      for e in mem.recent(3)])

    # 3) 随机回放：喂给世界模型的训练批
    batch = mem.sample(8)
    print("replay rewards:", [e.get("reward") for e in batch])

    # 4) 存取
    sd = mem.state_dict()
    mem2 = EpisodicMemory(capacity=1000, latent_dim=16)
    mem2.load_state_dict(sd)
    print("roundtrip:", len(mem) == len(mem2),
          "best-match same:", mem2.read(query).best.get("place") ==
          res.episodes[0].get("place"))
