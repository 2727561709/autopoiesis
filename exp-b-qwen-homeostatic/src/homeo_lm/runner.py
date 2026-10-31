"""智能体 Harness：统一的环境推进层 + 轨迹记录。
Agent harness: one shared rollout layer plus trajectory recording.

之前探针、PPO、L1 各写各的推进循环——三份几乎相同的代码，三个可以
各自漂移的地方：探针里改了观测格式，PPO 里没跟上，测出来的东西就不是
同一个策略。这一层把它们收成**一份**。
Previously probes, PPO and L1 each had their own rollout loop — three
near-identical copies, three places to drift apart: change the
observation format in the probe, forget to change it in PPO, and what
you measure is no longer the policy you trained. This layer collapses
them into ONE.

两条硬约束 / two hard constraints:
1. **必须批量**：一次决策 = 一次前向（首 token 分类），不是逐条
   `generate`。串行推进会把 Phase 0 拿到的约 200× 加速全还回去。
   It must be BATCHED: one decision = one forward pass (first-token
   classification), not one `generate` per sample. Serial rollouts
   would hand back the ~200x speed-up won in Phase 0.
2. **已结束的局不再前向**：每步只对还活着的局组批；跑完的留在轨迹里。
   Finished episodes get no further forward passes: each step batches
   only the still-alive ones; finished ones stay in the trace.

用法 / usage::

    runner = BatchRunner(model, tok, device, greedy=True)
    traces = runner.rollout(env_cfg, episodes=20, seed=9_000)
    print(runner.summary(traces))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .acthead import action_token_ids, greedy_with_legality
from .probe import prompt_ids
from .textenv import ACTION_WORDS, TextEnv


@dataclass
class Step:
    """一步的记录。One recorded step."""

    t: int
    action: int
    legal: bool
    energy: Optional[float] = None
    delta: Optional[float] = None
    done: bool = False
    died: bool = False


@dataclass
class Trace:
    """一局的完整轨迹。A whole episode as a trace."""

    seed: int
    steps: List[Step] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def illegal(self) -> int:
        return sum(1 for s in self.steps if not s.legal)

    @property
    def died(self) -> bool:
        return bool(self.steps and self.steps[-1].died)

    @property
    def survived(self) -> bool:
        return not self.died

    @property
    def illegal_rate(self) -> float:
        return self.illegal / max(1, self.length)


class BatchRunner:
    """批量推进 N 局（**一次前向 = 一批决策**）。
    Advance N episodes in batches (ONE forward pass = one batch of
    decisions).

    `driver` 决定谁来做动作：学生自己（默认，用首 token 分类）、教师、
    或随机。把它做成参数，是因为"on-policy 数据"和"教学数据"的差别
    就在这一步上（L1 的关键）。
    `driver` decides who acts: the student (default, first-token
    classification), the teacher, or random. It is a parameter because
    that single choice is the difference between on-policy data and
    teaching data — the crux of L1.
    """

    def __init__(self, model, tok, device: str = "cpu", greedy: bool = True,
                 batch_size: int = 32,
                 driver: Optional[Callable] = None) -> None:
        self.model, self.tok, self.device = model, tok, device
        self.greedy = greedy
        self.batch_size = batch_size
        self.act_ids = action_token_ids(tok)
        self.driver = driver

    def decide(self, prompts: Sequence[Sequence[int]]):
        """一批观测 → 一批动作（批量路径，一次前向）。
        Batch of observations -> batch of actions (batched, one forward)."""
        if self.driver is not None:
            return self.driver(prompts)
        acts, _ = greedy_with_legality(self.model, self.tok, prompts,
                                       self.act_ids, self.device,
                                       self.batch_size)
        return acts

    def rollout(self, env_cfg, episodes: int = 20, seed: int = 9_000,
                record: Optional[Callable] = None) -> List[Trace]:
        """并行推进 `episodes` 局，返回每局的轨迹。
        Advance `episodes` in parallel, returning one trace per episode.

        `record` 可选：每步的额外记录钩子（如能量 / 视野），由调用方决定
        要什么——这一层不替调用方猜指标。
        `record` is an optional per-step hook (energy, vision, ...); the
        caller decides what it wants. This layer does not guess metrics
        on the caller's behalf.
        """
        envs = [TextEnv(env_cfg) for _ in range(episodes)]
        for i, e in enumerate(envs):
            e.reset(seed=seed + i)
        traces = [Trace(seed=seed + i) for i in range(episodes)]
        alive = [True] * episodes
        while any(alive):
            idx = [i for i, a in enumerate(alive) if a]
            prompts = [prompt_ids(self.tok, envs[i].env,
                                  envs[i].prev_action) for i in idx]
            acts, legals = greedy_with_legality(
                self.model, self.tok, prompts, self.act_ids, self.device,
                self.batch_size)
            for i, a, lg in zip(idx, acts, legals):
                _, _, done, info = envs[i].step(ACTION_WORDS[a])
                extra = record(envs[i], i) if record else {}
                traces[i].steps.append(Step(
                    t=len(traces[i].steps), action=int(a), legal=bool(lg),
                    done=bool(done), died=bool(info.get("died")),
                    **extra))
                if done or len(traces[i].steps) >= getattr(
                        env_cfg, "max_steps", 400):
                    alive[i] = False
        return traces

    @staticmethod
    def summary(traces: Sequence[Trace]) -> Dict:
        """轨迹 → 汇总指标（存活率 / 平均步数 / 非法率）。
        Traces -> summary metrics (survival / mean steps / illegal rate)."""
        if not traces:
            return {"episodes": 0}
        lengths = [t.length for t in traces]
        steps = sum(lengths)
        return {
            "episodes": len(traces),
            "survival_rate": float(np.mean([t.survived for t in traces])),
            "mean_steps": float(np.mean(lengths)),
            "illegal_rate": sum(t.illegal for t in traces) / max(1, steps),
        }
