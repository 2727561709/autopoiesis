"""M16 规划器模块主实现。
M16 planner — main implementation.

职责：在 M10 世界模型的"脑内"做 CEM（交叉熵方法）想象规划 ——
采样动作序列 → imagine 评估 → 精英拟合 → 迭代收敛，返回最优序列。
Responsibilities: CEM (cross-entropy method) planning inside the M10
world model's imagination — sample action sequences -> evaluate via
imagine -> fit elites -> iterate; return the best sequence.

目标函数（默认纯回报最大化；goal 给定时为回报 + 终局接近目标）：
Objective (reward-only by default; with a goal, reward + terminal
goal-seeking term):
    score = Σ r_t  -  goal_weight * || z_H - goal ||₂

支持离散动作（每步独立类别分布）与连续动作（每步独立高斯）。
Supports discrete actions (per-step categorical distribution) and
continuous actions (per-step Gaussian).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import PlanResult, PlannerError

__all__ = ["Planner"]


class Planner:
    """基于世界模型的想象规划器。Imagination-based planner.

    Args:
        world_model: M10 WorldModel（或任何 forward(z, a) 兼容对象）。
            M10 WorldModel (or anything forward(z, a) compatible).
        action_dim: 动作维度（离散时 = 动作数）。Action dimension
            (= number of actions when discrete).
        discrete: True 时动作是 one-hot 类别；False 时连续向量。
            True for one-hot categorical actions; False for
            continuous vectors.
        horizon: 规划步数 H。Planning horizon H.
        population: 每代候选序列数。Candidates per generation.
        elite_frac: 精英比例 (0, 1]。Elite fraction in (0, 1].
        generations: CEM 迭代代数。CEM generation count.
        goal_weight: 终局目标距离的惩罚权重。Penalty weight of the
            terminal goal distance.
        seed: 随机种子。Random seed.
    """

    def __init__(self, world_model, action_dim: int, discrete: bool = True,
                 horizon: int = 8, population: int = 64, elite_frac: float = 0.25,
                 generations: int = 4, goal_weight: float = 1.0,
                 seed: int = 0) -> None:
        if action_dim <= 0:
            raise PlannerError("action_dim must be positive")
        if horizon <= 0:
            raise PlannerError("horizon must be positive")
        if population < 2:
            raise PlannerError("population must be >= 2")
        if not 0.0 < elite_frac <= 1.0:
            raise PlannerError("elite_frac must be in (0, 1]")
        if generations < 1:
            raise PlannerError("generations must be >= 1")
        self.wm = world_model
        self.action_dim = int(action_dim)
        self.discrete = bool(discrete)
        self.horizon = int(horizon)
        self.population = int(population)
        self.elite = max(1, int(population * elite_frac))
        self.generations = int(generations)
        self.goal_weight = float(goal_weight)
        self._rng = np.random.default_rng(seed)
        # 连续模式的分布参数 / continuous-mode distribution params
        self._mean = np.zeros((self.horizon, self.action_dim))
        self._std = np.ones((self.horizon, self.action_dim))

    # ------------------------------------------------------------------ API
    def plan(self, z: np.ndarray,
             goal: Optional[np.ndarray] = None) -> PlanResult:
        """在想象中搜索最优动作序列。
        Search for the best action sequence in imagination.

        Args:
            z: 当前潜状态 (latent_dim,)。Current latent state.
            goal: 可选目标潜状态 (latent_dim,)；给定时按
                `score = Σr - goal_weight * ||z_H - goal||` 优化。
                Optional goal latent state; when given, the objective
                becomes `score = Σr - goal_weight * ||z_H - goal||`.
        Returns:
            PlanResult（actions (H, action_dim) + 预测轨迹 + 得分）。
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 1:
            raise PlannerError(f"z must be 1-D, got {z.ndim}-D")
        if not np.all(np.isfinite(z)):
            raise PlannerError("z contains NaN/Inf")
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float64)
            if goal.ndim != 1:
                raise PlannerError("goal must be 1-D")
            if goal.shape[0] != z.shape[0]:
                raise PlannerError(
                    f"goal dim {goal.shape[0]} != latent dim {z.shape[0]}")

        # 离散：每步类别分布 (H, K) / discrete: per-step categorical
        probs = np.full((self.horizon, self.action_dim),
                        1.0 / self.action_dim)
        best: Optional[PlanResult] = None
        history: List[float] = []

        for _ in range(self.generations):
            seqs, scores = [], []
            for _p in range(self.population):
                if self.discrete:
                    ids = np.stack([
                        self._rng.choice(self.action_dim, p=probs[t])
                        for t in range(self.horizon)])
                    acts = self._onehot_batch(ids)
                else:
                    acts = np.clip(
                        self._mean + self._std * self._rng.standard_normal(
                            (self.horizon, self.action_dim)), -1.0, 1.0)
                res = self._rollout(z, acts, goal)
                seqs.append(res)
                scores.append(res.score)
            order = np.argsort(scores)[::-1]      # 分数降序 / descending
            elite_idx = order[:self.elite]
            history.append(float(scores[elite_idx[0]]))
            if best is None or scores[elite_idx[0]] > best.score:
                best = seqs[elite_idx[0]]
            # 精英拟合下一代分布 / fit next-gen distribution on elites
            if self.discrete:
                counts = np.zeros_like(probs)
                for i in elite_idx:
                    ids = seqs[i].actions.argmax(axis=-1)
                    counts[np.arange(self.horizon), ids] += 1.0
                # 拉普拉斯平滑防过早坍缩 / Laplace smoothing vs collapse
                probs = (counts + 1.0) / (self.elite + self.action_dim)
            else:
                elite_acts = np.stack([seqs[i].actions for i in elite_idx])
                self._mean = elite_acts.mean(axis=0)
                self._std = np.maximum(elite_acts.std(axis=0), 0.05)
        assert best is not None
        best.history = history
        return best

    __call__ = plan

    def reset(self) -> None:
        """重置连续模式的分布参数（换任务/换世界后调用）。
        Reset continuous-mode distribution params (call when the
        task or world changes)."""
        self._mean = np.zeros((self.horizon, self.action_dim))
        self._std = np.ones((self.horizon, self.action_dim))

    # -------------------------------------------------------------- internals
    def _rollout(self, z: np.ndarray, acts: np.ndarray,
                 goal: Optional[np.ndarray]) -> PlanResult:
        """用固定动作序列在世界模型里想象（M10 imagine 的序列版）。
        Imagine a fixed action sequence (the sequence form of M10's
        imagine)."""
        zs: List[np.ndarray] = [z.copy()]
        rewards: List[float] = []
        dones: List[float] = []
        cur = z
        for t in range(self.horizon):
            zp, r, d = self.wm(cur, acts[t])
            rewards.append(float(r))
            dones.append(float(d))
            zs.append(np.asarray(zp))
            cur = np.asarray(zp)
            if d > 0.5:
                # 预测终止：剩余步沿用最后动作，保持输出形状不变
                # predicted termination: pad remaining steps with the
                # last action to keep the output shape fixed
                rewards.extend([0.0] * (self.horizon - t - 1))
                dones.extend([1.0] * (self.horizon - t - 1))
                acts = np.repeat(acts[t][None], self.horizon, axis=0)
                break
        score = float(np.sum(rewards))
        if goal is not None:
            score -= self.goal_weight * float(
                np.linalg.norm(np.asarray(zs[-1]) - goal))
        return PlanResult(actions=acts, zs=zs, rewards=rewards,
                          dones=dones, score=score)

    def _onehot_batch(self, ids: np.ndarray) -> np.ndarray:
        out = np.zeros((self.horizon, self.action_dim))
        out[np.arange(self.horizon), ids] = 1.0
        return out
