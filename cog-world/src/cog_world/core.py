"""M26 世界环境模块主实现。

职责：2-D 竞技场上的食物/危险/资源生成与交互。
- 智能体（M25 Body）在世界中注册后占据一个位置。
- observe(id)：局部观测向量（位置归一化 + 食物气味 + 危险接近度 + 自身状态）。
- step(id, action)：身体执行动作 + 环境交互（吃到食物 -> eat 效果、
  踩到危险 -> damage），返回 (obs, reward)。
- reward（内在奖励雏形）：能量增量 + 完整性增量 - 危险惩罚 - 存活开销。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from cog_body import Body

from .types import Action, StepResult, WorldError

#: 观测向量维度（位置2 + 食物气味1 + 危险接近1 + 能量1 + 完整性1）
OBS_DIM = 6
#: 吃到食物的判定半径
EAT_RADIUS = 0.5
#: 踩到危险的判定半径与伤害
DANGER_RADIUS = 0.5
DANGER_DAMAGE = 0.1
#: 每步存活开销（鼓励高效行为）
LIVING_COST = 0.01


class World:
    """2-D 网格世界（连续坐标）。

    Args:
        size: 竞技场边长。
        n_food: 食物数量。
        n_danger: 危险区域数量。
        seed: 随机种子（决定食物/危险布局，可复现）。
    """

    def __init__(self, size: float = 10.0, n_food: int = 10, n_danger: int = 3,
                 seed: int = 0) -> None:
        if size <= 0 or n_food < 0 or n_danger < 0:
            raise WorldError("size must be > 0 and counts must be >= 0")
        self.size = float(size)
        self.n_food = int(n_food)
        self.n_danger = int(n_danger)
        self.agents: Dict[int, Body] = {}
        self._next_id = 0
        self.reset(seed)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> None:
        """重新生成食物与危险布局（智能体保留）。"""
        rng = np.random.default_rng(seed if seed is not None else 0)
        self._rng = rng
        self.food: List[Tuple[float, float]] = [
            (float(rng.uniform(0, self.size)), float(rng.uniform(0, self.size)))
            for _ in range(self.n_food)
        ]
        self.dangers: List[Tuple[float, float]] = [
            (float(rng.uniform(0, self.size)), float(rng.uniform(0, self.size)))
            for _ in range(self.n_danger)
        ]

    def add_agent(self, body: Body | None = None, position=None) -> int:
        """注册一个智能体身体，返回其 ID。"""
        if body is None:
            pos = tuple(position) if position is not None else (self.size / 2,) * 2
            body = Body(position=pos, arena_size=self.size)
        elif position is not None:
            raise WorldError("provide either body or position, not both")
        aid = self._next_id
        self._next_id += 1
        self.agents[aid] = body
        return aid

    def observe(self, agent_id: int) -> np.ndarray:
        """局部观测（OBS_DIM,）：归一化位置、食物气味、危险接近、自身状态。"""
        body = self._get(agent_id)
        p = np.array(body.position) / self.size
        return np.concatenate([
            p,
            [self._food_scent(body.position)],
            [self._danger_proximity(body.position)],
            [body.energy],
            [body.integrity],
        ])

    def step(self, agent_id: int, action: Action) -> StepResult:
        """智能体执行一步：身体动作 + 环境交互。

        Returns:
            (obs, reward)
        """
        body = self._get(agent_id)
        pos_before = body.position

        # 1. 身体执行动作（不含 eat —— 吃多少由环境判定）
        env_action = dict(action)
        env_action.pop("eat", None)
        effects = body.step(env_action)

        # 2. 环境交互：吃到附近食物
        eaten = self._try_eat(body.position)
        if eaten > 0:
            body.step({"eat": eaten})
            effects["energy"] = effects.get("energy", 0.0) + eaten

        # 3. 踩到危险
        if self._danger_proximity(body.position) > 0.99:
            dmg = body.step({"damage": DANGER_DAMAGE})
            effects["integrity"] = effects.get("integrity", 0.0) + dmg["integrity"]

        # 4. 内在奖励雏形：趋利避害 + 存活开销
        reward = (
            effects.get("energy", 0.0)
            + effects.get("integrity", 0.0)
            - LIVING_COST
        )

        obs = self.observe(agent_id)
        return obs, float(reward)

    # ------------------------------------------------------------------ 内部
    def _get(self, agent_id: int) -> Body:
        if agent_id not in self.agents:
            raise WorldError(f"unknown agent id: {agent_id!r}")
        return self.agents[agent_id]

    def _try_eat(self, pos: Tuple[float, float]) -> float:
        """吃掉判定半径内的第一份食物，返回能量增量（0 表示没吃到）。"""
        for i, (fx, fy) in enumerate(self.food):
            if abs(fx - pos[0]) <= EAT_RADIUS and abs(fy - pos[1]) <= EAT_RADIUS:
                self.food.pop(i)
                return 0.3  # 一份食物的能量
        return 0.0

    def _food_scent(self, pos: Tuple[float, float]) -> float:
        """食物气味 ∈ [0,1]：最近食物距离的负指数。"""
        if not self.food:
            return 0.0
        d = min((fx - pos[0]) ** 2 + (fy - pos[1]) ** 2 for fx, fy in self.food) ** 0.5
        return float(np.exp(-d / self.size))

    def _danger_proximity(self, pos: Tuple[float, float]) -> float:
        """危险接近度 ∈ [0,1]：进入判定半径内为 1，否则随距离衰减。"""
        best = 0.0
        for dx, dy in self.dangers:
            d = ((dx - pos[0]) ** 2 + (dy - pos[1]) ** 2) ** 0.5
            if d <= DANGER_RADIUS:
                return 1.0
            best = max(best, float(np.exp(-(d - DANGER_RADIUS) / self.size)))
        return best
