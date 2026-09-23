"""M26 最小可运行示例：python examples/demo.py"""
import numpy as np

from cog_world import World

if __name__ == "__main__":
    world = World(size=10, n_food=10, n_danger=3, seed=0)
    aid = world.add_agent()

    total = 0.0
    rng = np.random.default_rng(0)
    for t in range(20):
        action = {"move": tuple(rng.uniform(-0.5, 0.5, size=2))}
        obs, reward = world.step(aid, action)
        total += reward
        print(f"t={t:2d} obs={obs.round(2)} reward={reward:+.3f}")

    print("总奖励:", round(total, 3), "剩余食物:", len(world.food))
