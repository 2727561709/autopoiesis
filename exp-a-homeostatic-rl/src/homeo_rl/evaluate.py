"""阶段 A 行为探针：验证"生命性"是否涌现。
Stage-A behavioral probes: test whether "vitality" emerges.

    探针1 存活：随机初始条件下贪心策略能活多久、稳态维持如何
    Probe 1 — Survival: how long the greedy policy lives from random
    initial conditions, and how well it maintains homeostasis.

    探针2 觅食（两档）：
    Probe 2 — Foraging (two tiers):
        near_visible —— 低能量 + 食物可见（2 格），测目标追击；
        near_visible — low energy + food visible (2 cells): tests
        goal-directed pursuit;
        far_search   —— 食物在视野外（4 格随机方向）+ 足够能量预算，测搜索
        far_search   — food out of view (4 cells, random direction) with
        a sufficient energy budget: tests search.

    探针3 方向性与调制：见 direction_probe
    Probe 3 — Directionality & modulation: see direction_probe.

若模型真的"为生存而学"，应当看到：
If the model truly "learns in order to survive", we expect:
    - 存活率随训练上升，能量维持在设定点附近；
      survival rises with training and energy stays near the setpoint;
    - 觅食效率显著高于随机策略；
      foraging efficiency well above the random policy;
    - 预测误差随训练下降（世界模型越来越准）。
      prediction error falls with training (a better world model).

用法 / usage:
    python -m homeo_rl.evaluate --ckpt runs/exp-a/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .env import EnvConfig, HomeostaticGridWorld, N_ACTIONS
from .model import HomeoActorCritic
from .train import NONE_ACTION


def load_model(ckpt_path: Path, device: str = "cpu") -> tuple[HomeoActorCritic, EnvConfig]:
    """载入 checkpoint，返回 (模型, 训练时的环境配置)。
    Load a checkpoint; returns (model, env config from training)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    env_cfg = EnvConfig(**ckpt["env_cfg"])
    model = HomeoActorCritic(view=env_cfg.view).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, env_cfg


def _act_greedy(model: HomeoActorCritic, env: HomeostaticGridWorld,
                prev_a: int, device: str) -> int:
    """贪心单步动作（确定性 argmax）。One greedy (argmax) step."""
    v = torch.from_numpy(env.last_obs["vision"]).unsqueeze(0).to(device)
    i = torch.from_numpy(env.last_obs["intero"]).unsqueeze(0).to(device)
    pa = torch.tensor([prev_a], dtype=torch.long, device=device)
    a, _, _ = model.act(v, i, pa, deterministic=True)
    return int(a.item())


# ------------------------------------------------------- 探针 1：存活 / survival
def survival_probe(model: HomeoActorCritic, env_cfg: EnvConfig,
                   episodes: int = 20, device: str = "cpu") -> Dict:
    """贪心策略完整回合：步数、死因分布、终末能量/完整性。
    Full greedy episodes: step counts, death distribution, final
    energy/integrity."""
    lens, energies, integrities, deaths = [], [], [], 0
    for ep in range(episodes):
        env = HomeostaticGridWorld(EnvConfig(
            **{**vars(env_cfg), "seed": 9_000 + ep}))
        env.reset()
        prev_a = NONE_ACTION
        while True:
            a = _act_greedy(model, env, prev_a, device)
            _, r, done, info = env.step(a)
            prev_a = a
            if done:
                lens.append(env.t)
                energies.append(info["energy"])
                integrities.append(info["integrity"])
                deaths += int(info["died"])
                break
    return {
        "episodes": episodes,
        "mean_steps": float(np.mean(lens)),
        "max_steps_cap": env_cfg.max_steps,
        "survival_rate": 1.0 - deaths / episodes,
        "mean_final_energy": float(np.mean(energies)),
        "mean_final_integrity": float(np.mean(integrities)),
    }


# ------------------------------------------------------- 探针 2：觅食 / foraging
# 方向 -> (行位移, 列位移) / direction -> (row delta, col delta)
_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _forage_once(policy, env_cfg: EnvConfig, seed: int, distance: int,
                 energy: float, horizon: int, direction: str,
                 device: str = "cpu") -> Dict:
    """受控觅食场景：清空危险，食物放在指定方向 distance 格处。
    Controlled foraging scenario: hazards cleared, one food cell placed
    `distance` cells away in `direction`."""
    env = HomeostaticGridWorld(EnvConfig(**{**vars(env_cfg), "seed": seed}))
    env.reset()
    n = env.n
    r, c = env.agent
    env.hazards = set()
    dr, dc = _DIRS[direction]
    env.foods = {(min(max(r + dr * distance, 0), n - 1),
                  min(max(c + dc * distance, 0), n - 1))}
    env.energy = energy
    env.integrity = 0.9
    env.t = 0
    env.last_obs = env.observation()
    prev_a = NONE_ACTION
    for t in range(1, horizon + 1):
        a = policy(env, prev_a)
        _, rew, done, info = env.step(a)
        prev_a = a
        if info["ate"]:
            return {"success": True, "steps": t}
        if done:
            break
    return {"success": False, "steps": horizon}


# (名称, 距离, 起手能量, 步数上限)：near 可见测追击；far 不可见测搜索
# (tag, distance, start energy, horizon): near = visible, tests pursuit;
# far = out of view, tests search.
_PROBE_VARIANTS = (
    ("near_visible", 2, 0.3, 30),
    ("far_search", 4, 0.5, 40),
)


def foraging_probe(model: Optional[HomeoActorCritic], env_cfg: EnvConfig,
                   trials: int = 50, device: str = "cpu") -> Dict:
    """两档觅食探针，model=None 时用随机策略做基线。
    Two-tier foraging probe; a random policy baseline when model=None."""
    if model is not None:
        policy = lambda env, pa: _act_greedy(model, env, pa, device)
        name = "trained"
    else:
        rng0 = np.random.default_rng(0)
        policy = lambda env, pa: int(rng0.integers(N_ACTIONS))
        name = "random"
    out: Dict = {"policy": name, "trials": trials}
    for tag, distance, energy, horizon in _PROBE_VARIANTS:
        # 两个策略用同一方向序列，配对比较 / both policies share the
        # same direction sequence for a paired comparison
        dir_rng = np.random.default_rng(1234)
        dirs = list(_DIRS)
        results = [
            _forage_once(policy, env_cfg, seed=70_000 + k, distance=distance,
                         energy=energy, horizon=horizon,
                         direction=dirs[int(dir_rng.integers(4))])
            for k in range(trials)
        ]
        out[tag] = {
            "distance": distance,
            "start_energy": energy,
            "horizon": horizon,
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_steps_to_food": float(np.mean([r["steps"] for r in results])),
        }
    return out


# ---------------------------------- 探针 3：方向性与内稳态调制 / directionality & modulation
def direction_probe(model: HomeoActorCritic, env_cfg: EnvConfig,
                    trials: int = 20, device: str = "cpu") -> Dict:
    """三个定向性行为检查（回合起点恒在网格中心，无越界问题）：
    Three directionality checks (episodes always start at the grid
    center, so no out-of-bounds issues):

    1. food_approach      —— 低能量 + 食物可见于某方向 3 格，应朝食物移动；
    1. food_approach      — low energy + food visible 3 cells away: the
    agent should move toward the food;
    2. satiated_ignores   —— 能量在设定点上 + 食物可见，应不再朝食物移动
                             （行为被内稳态状态调制的直接证据）；
    2. satiated_ignores   — energy at setpoint + food visible: the agent
    should NOT move toward the food (direct evidence that behavior is
    modulated by homeostatic state);
    3. hazard_avoidance   —— 危险紧邻时，不应移入危险格。
    3. hazard_avoidance   — hazard adjacent: the agent must not step into
    the hazard cell.
    """
    center_probe_base = 60_000

    def _toward_action(dr: int, dc: int) -> int:
        """朝目标方向应执行的动作。The action that moves toward (dr, dc)."""
        if abs(dr) >= abs(dc):
            return 1 if dr > 0 else 0
        return 3 if dc > 0 else 2

    def _run(place, energy: float, integrity: float) -> Dict[int, int]:
        """构造场景并统计首步动作分布。Build the scenario and tally the
        first action across trials."""
        counts: Dict[int, int] = {}
        for t in range(trials):
            env = HomeostaticGridWorld(EnvConfig(
                **{**vars(env_cfg), "seed": center_probe_base + t}))
            env.reset()
            r, c = env.agent
            env.hazards, env.foods = set(), set()
            place(env, r, c)
            env.energy, env.integrity, env.t = energy, integrity, 0
            env.last_obs = env.observation()
            a = _act_greedy(model, env, NONE_ACTION, device)
            counts[a] = counts.get(a, 0) + 1
        return counts

    offsets = {"up": (-3, 0), "down": (3, 0), "left": (0, -3), "right": (0, 3)}
    out: Dict = {"trials": trials}

    # 1) 低能量定向接近 / goal-directed approach under low energy
    approach = {}
    for name, (dr, dc) in offsets.items():
        counts = _run(lambda e, r, c, dr=dr, dc=dc: e.foods.add((r + dr, c + dc)),
                      energy=0.3, integrity=0.9)
        approach[name] = counts.get(_toward_action(dr, dc), 0) / trials
    out["food_approach_rate"] = approach

    # 2) 饱腹时无视食物（越低越好）/ ignore food when satiated (lower is better)
    counts = _run(lambda e, r, c: e.foods.add((r, c + 3)),  # 食物在右侧 / food to the right
                  energy=env_cfg.energy_setpoint, integrity=0.9)
    out["satiated_approach_rate"] = counts.get(3, 0) / trials  # 越低越好 / lower is better

    # 3) 危险避让（越低越好）/ hazard avoidance (lower is better)
    counts = _run(lambda e, r, c: e.hazards.add((r, c + 1)),
                  energy=0.7, integrity=0.85)
    out["hazard_entry_rate"] = counts.get(3, 0) / trials  # 越低越好 / lower is better
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="阶段A行为探针 / Stage-A behavioral probes")
    ap.add_argument("--ckpt", type=str, default="runs/exp-a/checkpoint.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    model, env_cfg = load_model(ckpt_path)
    report = {
        "checkpoint": str(ckpt_path),
        "survival": survival_probe(model, env_cfg, args.episodes),
        "foraging": {
            "trained": foraging_probe(model, env_cfg, args.trials),
            "random_baseline": foraging_probe(None, env_cfg, args.trials),
        },
        "direction_modulation": direction_probe(model, env_cfg),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
