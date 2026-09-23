"""内稳态网格世界：阶段 A 的生存环境。
Homeostatic grid world: the survival environment for Stage A.

一个"最小生命体"在网格中生存：
A "minimal organism" survives on a grid:
    - 能量持续衰减（代谢），移动比休息消耗更多；
      Energy decays continuously (metabolism); moving costs more than resting;
    - 食物恢复能量，吃完在别处重生；
      Food restores energy and respawns elsewhere once eaten;
    - 危险区损伤完整性，完整性缓慢自愈；
      Hazard cells damage integrity; integrity heals slowly;
    - 能量或完整性归零即死亡。
      Energy or integrity reaching zero means death.

观察 = 自我中心的局部视野 (k, k, 4) + 内感受 (能量, 完整性)。
Observation = ego-centric local view (k, k, 4) + interoception
(energy, integrity).
没有任何外部任务 —— 唯一的回报来自内稳态缺口的缩减（见 drives.py）。
There are NO external tasks — the only reward comes from homeostatic
deficit reduction (see drives.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .drives import total_deficit

__all__ = ["ACTION_NAMES", "N_ACTIONS", "EnvConfig", "HomeostaticGridWorld"]

N_ACTIONS = 5
ACTION_NAMES = ("up", "down", "left", "right", "rest")
# 动作 -> (行位移, 列位移) / action -> (row delta, col delta)
_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


@dataclass
class EnvConfig:
    """环境超参数。Environment hyperparameters."""

    grid_size: int = 12
    view: int = 7
    n_food: int = 8
    n_hazard: int = 6
    food_energy: float = 0.35
    move_cost: float = 0.015
    rest_cost: float = 0.005
    hazard_damage: float = 0.15
    heal_rate: float = 0.002
    max_steps: int = 1000
    init_energy: float = 0.7
    init_integrity: float = 0.85
    energy_setpoint: float = 0.8
    integrity_setpoint: float = 0.9
    death_penalty: float = -3.0
    seed: Optional[int] = None
    # 休克模式：能量归零不死，昏迷一段时间（大惩罚+时间损失，保留不可逆性）。
    # 对照实验：检验"存在终结"本身是否必要，还是"大惩罚+时间损失"就够。
    # Coma mode: hitting zero energy does not kill; the agent falls into a
    # coma for a while (big penalty + time loss, keeping irreversibility).
    # Control experiment: is "existential termination" itself necessary,
    # or are "big penalty + time loss" enough?
    coma_mode: bool = False
    coma_duration: int = 50    # 昏迷步数 / coma duration in steps
    coma_regen: float = 0.004  # 昏迷中每步能量回升（50 步共 +0.2）/ energy regenerated per coma step (+0.2 over 50 steps)
    coma_penalty: float = -1.5  # 陷入昏迷的一次性惩罚 / one-time penalty for falling into a coma


class HomeostaticGridWorld:
    """单个内稳态生存环境实例。
    A single homeostatic survival environment instance.

    注意：`agent`、`foods`、`hazards`、`energy`、`integrity` 是公开属性，
    评测探针可以直接操纵它们来构造受控场景。
    Note: `agent`, `foods`, `hazards`, `energy`, `integrity` are public
    attributes — evaluation probes may manipulate them directly to build
    controlled scenarios.
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.cfg = config or EnvConfig()
        self.n = self.cfg.grid_size
        self._episode = 0
        self._rng = np.random.default_rng(self.cfg.seed)
        self.last_obs: Optional[Dict[str, np.ndarray]] = None
        self.reset()

    # --------------------------------------------------------- 生命周期 / lifecycle
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """重置环境。不传 seed 时按内部计数器派生，保证各回合独立随机。
        Reset the environment. Without an explicit seed, one is derived
        from an internal counter so every episode is independently random."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self.cfg.seed is not None:
            self._rng = np.random.default_rng(self.cfg.seed + self._episode)
        self._episode += 1
        self.t = 0
        self.agent = (self.n // 2, self.n // 2)
        self.energy = float(self.cfg.init_energy)
        self.integrity = float(self.cfg.init_integrity)
        self.coma_steps_left = 0
        self._place_objects()
        self.last_obs = self._observation()
        return self.last_obs

    def _place_objects(self) -> None:
        """随机放置危险与食物（互不重叠，不压在智能体上）。
        Randomly place hazards and food (non-overlapping, never on the agent)."""
        n = self.n
        cells = [(r, c) for r in range(n) for c in range(n) if (r, c) != self.agent]
        self._rng.shuffle(cells)
        k = self.cfg.n_hazard
        self.hazards: Set[Tuple[int, int]] = set(cells[:k])
        self.foods: Set[Tuple[int, int]] = set(cells[k:k + self.cfg.n_food])

    def _respawn_food(self) -> None:
        """吃掉的食物在随机空格重生，保持食物总数恒定。
        Respawn the eaten food at a random free cell, keeping the food
        count constant."""
        occupied = self.hazards | self.foods | {self.agent}
        while True:
            cell = (int(self._rng.integers(self.n)), int(self._rng.integers(self.n)))
            if cell not in occupied:
                self.foods.add(cell)
                return

    # ------------------------------------------------------------ 动力学 / dynamics
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """执行动作，返回 (obs, reward, done, info)。
        Apply an action; returns (obs, reward, done, info).

        reward = 内稳态缺口缩减量（死亡时叠加死亡惩罚）。
        reward = homeostatic deficit reduction (plus the death penalty
        on death, or the coma penalty on coma onset).
        """
        cfg = self.cfg
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"invalid action: {action!r}")

        setpoints = (cfg.energy_setpoint, cfg.integrity_setpoint)
        before = (self.energy, self.integrity)
        d_before = total_deficit(before, setpoints)

        was_coma = self.coma_steps_left > 0
        if was_coma:
            # 昏迷中：动作被覆盖（原地不动），能量缓慢回升
            # In a coma: the action is overridden (no movement); energy
            # regenerates slowly.
            self.coma_steps_left -= 1
            self.energy = float(np.clip(self.energy + cfg.coma_regen, 0.0, 1.0))
            ate = False
        else:
            # 移动（撞墙原地不动，但仍付移动成本）或休息
            # Move (bumping a wall stays in place but still pays the move
            # cost) or rest.
            if action < 4:
                dr, dc = _DELTAS[action]
                r, c = self.agent
                self.agent = (min(max(r + dr, 0), self.n - 1),
                              min(max(c + dc, 0), self.n - 1))
            cost = cfg.rest_cost if action == 4 else cfg.move_cost
            self.energy = float(np.clip(self.energy - cost, 0.0, 1.0))

            # 进食 / eating
            ate = self.agent in self.foods
            if ate:
                self.foods.discard(self.agent)
                self.energy = float(np.clip(self.energy + cfg.food_energy, 0.0, 1.0))
                self._respawn_food()

        # 自愈与危险（先自愈后受伤：保证受伤可以把完整性打到 0 = 死亡可达）
        # Healing and hazards (heal BEFORE damage: this guarantees damage
        # can drive integrity to exactly 0, i.e. death is reachable).
        self.integrity = float(np.clip(self.integrity + cfg.heal_rate, 0.0, 1.0))
        hurt = self.agent in self.hazards
        if hurt:
            self.integrity = float(np.clip(self.integrity - cfg.hazard_damage, 0.0, 1.0))

        entering_coma = False
        if self.energy <= 0.0 and self.integrity > 0.0 and cfg.coma_mode:
            # 能量耗尽 → 休克而非死亡（完整性归零仍然死亡）
            # Energy exhausted -> coma instead of death (zero integrity
            # still kills).
            self.coma_steps_left = cfg.coma_duration
            entering_coma = True
        died = (self.energy <= 0.0 or self.integrity <= 0.0) and not entering_coma
        self.t += 1
        done = died or self.t >= cfg.max_steps

        d_after = total_deficit((self.energy, self.integrity), setpoints)
        reward = d_before - d_after
        if died:
            reward += cfg.death_penalty
        if entering_coma:
            reward += cfg.coma_penalty

        info = {
            "energy": self.energy,
            "integrity": self.integrity,
            "ate": ate,
            "hurt": hurt,
            "died": died,
            "coma": entering_coma or was_coma,
            "t": self.t,
        }
        self.last_obs = self._observation()
        return self.last_obs, float(reward), bool(done), info

    # ------------------------------------------------------------ 观察 / observation
    def observation(self) -> Dict[str, np.ndarray]:
        """当前观察（公开接口，评测探针在操纵内部状态后可重新渲染）。
        Current observation (public API; probes can re-render after
        manipulating internal state)."""
        return self._observation()

    def _observation(self) -> Dict[str, np.ndarray]:
        """自我中心视野：通道 0=边界 1=食物 2=危险 3=自身。
        Ego-centric view: channel 0=boundary 1=food 2=hazard 3=self."""
        k = self.cfg.view
        off = k // 2
        r, c = self.agent
        v = np.zeros((k, k, 4), dtype=np.float32)
        for dr in range(-off, off + 1):
            for dc in range(-off, off + 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.n and 0 <= cc < self.n):
                    v[dr + off, dc + off, 0] = 1.0  # 边界 / boundary
                else:
                    if (rr, cc) in self.foods:
                        v[dr + off, dc + off, 1] = 1.0
                    if (rr, cc) in self.hazards:
                        v[dr + off, dc + off, 2] = 1.0
        v[off, off, 3] = 1.0  # 自身 / self
        intero = np.array([self.energy, self.integrity], dtype=np.float32)
        return {"vision": v, "intero": intero}
