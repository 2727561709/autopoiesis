"""M16 最小可运行示例：python examples/demo.py

用内置 stub 世界模型演示；换成 M10 WorldModel 用法完全相同。
Demo with the built-in stub world model; drop in an M10 WorldModel
the same way.
"""
import numpy as np

from cog_planner import Planner


class ToyWM:
    """z' = 0.5 * roll(z)；动作 0 高回报。action 0 pays well."""

    def __call__(self, z, a):
        a = np.asarray(a, dtype=np.float64)
        return 0.5 * np.roll(z, 1), (0.8 if int(a.argmax()) == 0 else -0.2), 0.02


if __name__ == "__main__":
    pl = Planner(ToyWM(), action_dim=3, horizon=6, population=48,
                 generations=5, seed=0)
    z = np.array([0.5, -0.5, 0.2, -0.2])

    # 1) 纯回报规划
    res = pl.plan(z)
    print("actions :", res.actions.argmax(axis=-1))
    print("score   :", round(res.score, 3),
          " history:", [round(h, 2) for h in res.history])
    print("first   :", res.first_action.argmax(), " (MPC 只执行第一步)")

    # 2) 带目标的规划：把终局拉向 -z
    goal = -z
    res2 = pl.plan(z, goal=goal)
    print("goal dir score:", round(res2.score, 3),
          " end-dist:", round(float(np.linalg.norm(res2.zs[-1] - goal)), 3))
