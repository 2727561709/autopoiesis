"""测试用 stub 世界模型（保持模块独立测试原则，不依赖 M10）。
Stub world model for tests (keeps modules independently tested; no
M10 dependency).

规则与 M10 测试中的玩具环境一致：z' = 0.5 * roll(z)；argmax(a)==0
时 r=+0.8，否则 r=-0.2；done≈0.018（非终止先验）。
Same rules as the toy env in M10 tests: z' = 0.5 * roll(z); r=+0.8
when argmax(a)==0 else r=-0.2; done ≈ 0.018 (non-terminal prior).
"""
import numpy as np


class StubWM:
    def __init__(self, latent_dim: int = 4, action_dim: int = 2,
                 done: float = 0.018) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self._done = done
        self.calls = 0

    def __call__(self, z, a):
        self.calls += 1
        a = np.asarray(a, dtype=np.float64)
        z_next = 0.5 * np.roll(z, 1)
        r = 0.8 if int(a.argmax()) == 0 else -0.2
        return z_next, r, self._done

    forward = __call__
