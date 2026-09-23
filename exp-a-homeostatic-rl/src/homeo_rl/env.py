"""内稳态网格世界：阶段 A 的生存环境。

一个"最小生命体"在网格中生存：
    - 能量持续衰减（代谢），移动比休息消耗更多；
    - 食物恢复能量，吃完在别处重生；
    - 危险区损伤完整性，完整性缓慢自愈；
    - 能量或完整性归零即死亡。

观察 = 自我中心的局部视野 (k, k, 4) + 内感受 (能量, 完整性)。
没有任何外部任务 —— 唯一的回报来自内稳态缺口的缩减（见 drives.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Set, Tuple

import numpy as np

from .drives import total_deficit

__all__ = ["ACTION_NAMES", "N_ACTIONS", "EnvConfig", "HomeostaticGridWorld"]

N_ACTIONS = 5
ACTION_NAMES = ("up", "down", "left", "right", "rest")
_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


@dataclass
class EnvConfig:
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
    coma_mode: bool = False
    coma_duration: int = 50    # 昏迷步数
    coma_regen: float = 0.004  # 昏迷中每步能量回升（50 步共 +0.2）
    coma_penalty: float = -1.5  # 陷入昏迷的一次性惩罚


class HomeostaticGridWorld:
    """单个内稳态生存环境实例。

    注意：`agent`、`foods`、`hazards`、`energy`、`integrity` 是公开属性，
    评测探针可以直接操纵它们来构造受控场景。
    """

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.cfg = config or EnvConfig()
        self.n = self.cfg.grid_size
        self._episode = 0
        self._rng = np.random.default_rng(self.cfg.seed)
        self.last_obs: Optional[Dict[str, np.ndarray]] = None
        self.reset()

    # ------------------------------------------------------------------ 生命周期
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """重置环境。不传 seed 时按内部计数器派生，保证各回合独立随机。"""
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
        n = self.n
        cells = [(r, c) for r in range(n) for c in range(n) if (r, c) != self.agent]
        self._rng.shuffle(cells)
        k = self.cfg.n_hazard
        self.hazards: Set[Tuple[int, int]] = set(cells[:k])
        self.foods: Set[Tuple[int, int]] = set(cells[k:k + self.cfg.n_food])

    def _respawn_food(self) -> None:
        occupied = self.hazards | self.foods | {self.agent}
        while True:
            cell = (int(self._rng.integers(self.n)), int(self._rng.integers(self.n)))
            if cell not in occupied:
                self.foods.add(cell)
                return

    # ------------------------------------------------------------------ 动力学
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """执行动作，返回 (obs, reward, done, info)。

        reward = 内稳态缺口缩减量（死亡时叠加死亡惩罚）。
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
            self.coma_steps_left -= 1
            self.energy = float(np.clip(self.energy + cfg.coma_regen, 0.0, 1.0))
            ate = False
        else:
            # 移动（撞墙原地不动，但仍付移动成本）或休息
            if action < 4:
                dr, dc = _DELTAS[action]
                r, c = self.agent
                self.agent = (min(max(r + dr, 0), self.n - 1),
                              min(max(c + dc, 0), self.n - 1))
            cost = cfg.rest_cost if action == 4 else cfg.move_cost
            self.energy = float(np.clip(self.energy - cost, 0.0, 1.0))

            # 进食
            ate = self.agent in self.foods
            if ate:
                self.foods.discard(self.agent)
                self.energy = float(np.clip(self.energy + cfg.food_energy, 0.0, 1.0))
                self._respawn_food()

        # 自愈与危险（先自愈后受伤：保证受伤可以把完整性打到 0 = 死亡可达）
        self.integrity = float(np.clip(self.integrity + cfg.heal_rate, 0.0, 1.0))
        hurt = self.agent in self.hazards
        if hurt:
            self.integrity = float(np.clip(self.integrity - cfg.hazard_damage, 0.0, 1.0))

        entering_coma = False
        if self.energy <= 0.0 and self.integrity > 0.0 and cfg.coma_mode:
            # 能量耗尽 → 休克而非死亡（完整性归零仍然死亡）
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

    # ------------------------------------------------------------------ 观察
    def observation(self) -> Dict[str, np.ndarray]:
        """当前观察（公开接口，评测探针在操纵内部状态后可重新渲染）。"""
        return self._observation()

    def _observation(self) -> Dict[str, np.ndarray]:
        k = self.cfg.view
        off = k // 2
        r, c = self.agent
        v = np.zeros((k, k, 4), dtype=np.float32)
        for dr in range(-off, off + 1):
            for dc in range(-off, off + 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.n and 0 <= cc < self.n):
                    v[dr + off, dc + off, 0] = 1.0  # 边界
                else:
                    if (rr, cc) in self.foods:
                        v[dr + off, dc + off, 1] = 1.0
                    if (rr, cc) in self.hazards:
                        v[dr + off, dc + off, 2] = 1.0
        v[off, off, 3] = 1.0  # 自身
        intero = np.array([self.energy, self.integrity], dtype=np.float32)
        return {"vision": v, "intero": intero}
