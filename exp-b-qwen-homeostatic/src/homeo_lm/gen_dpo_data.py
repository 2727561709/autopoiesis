"""同状态穷举对比（+势函数整形）→ DPO 偏好对数据集。
Same-state exhaustive contrast (+ potential shaping) -> DPO pairs.

v2 的教训：SFT 行为克隆能拟合动作分布，却把"内感受→行动"的因果学反了。
v3 换 DPO，且偏好对不用教师标签，而用**环境真实的一步内稳态后果**。

裸一步对比的坑（冒烟测试发现，2026-10-15）：内稳态奖励是稀疏的——
食物 3 格远时，朝食物走与乱走的一步回报完全相同（都是移动成本 -0.015），
只有 rest 最优（-0.005）。裸对比会产出"永远休息"的坍缩标签。

修复：**势函数整形（PBRS）**——
    shaped(s, a) = r(s,a) + w · [φ(s') - φ(s)]
    φ(s) = -dist(agent, nearest food)    仅当饥饿时激活（门控）
         + +dist(agent, nearest hazard)  仅当受伤时激活（门控）
朝食物 1 步 → +w；远离 → -w；饱腹 → 门控关闭，rest 最优。
PBRS 不改变最优策略（只把稀疏信号变稠密），且**内感受调制直接内建于
门控**——饥饿时 chosen=朝食物，饱腹时 chosen=rest，正是 probe 的期望。

The v2 lesson: SFT clones behavior but learns the interoception ->
action causation REVERSED. v3 switches to DPO with pairs from the
environment's true one-step homeostatic consequences.

The pitfall of the naive one-step contrast (found in the smoke test,
2026-10-15): homeostatic reward is SPARSE — with food 3 cells away,
walking toward it costs the same -0.015 as walking anywhere, and rest
always wins. Naive pairs would collapse to "always rest".

Fix: **potential-based reward shaping (PBRS)** —
    shaped(s, a) = r(s,a) + w · [φ(s') - φ(s)]
    φ(s) = -dist(agent, nearest food)    gated on by hunger
         + +dist(agent, nearest hazard)  gated on by injury
Toward food = +w; away = -w; when satiated the gate closes and rest
wins. PBRS provably does not change the optimal policy (it only
densifies the sparse signal), and the interoceptive modulation is
built directly into the gate — hungry: chosen=toward food; satiated:
chosen=rest. Exactly what the probes expect.

用法 / usage:
    python -m homeo_lm.gen_dpo_data --teacher ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt --pairs 4000 --out data/dpo_pairs.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from homeo_rl.env import EnvConfig, N_ACTIONS

from .textenv import ACTION_WORDS, TextEnv, obs_messages

# torch / 教师模型只在真正生成数据时才导入——势函数整形的纯逻辑
# （potential / decide_gates / rank_actions_at_state）因此可以脱离
# torch 被单元测试直接覆盖。
# torch and the teacher model are imported lazily, inside the functions
# that need them, so the pure shaping logic can be unit-tested without
# pulling in torch at all.

NONE_ACTION = N_ACTIONS


def _manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def potential(env, gates: Optional[Dict[str, bool]] = None,
              gate_margin: float = 0.05) -> float:
    """内感受门控的导航势函数（门控在采样状态一次性判定）。
    Interoception-gated navigation potential (gates are decided ONCE
    at the sampled state).

    φ = -dist(nearest food)    （饥饿门开时）
        + dist(nearest hazard)（受伤门开时）

    视野连续性（2026-10-15 冒烟测试发现的第三个坑）：若把"可见性"
    直接写进 φ，食物在视野边缘时向外走 1 步会让它"离开视野"、φ 跳
    回 0，反而拿到正整形——经典的 PBRS 不连续陷阱。修复：门控在
    采样状态 s 判定（食物在 s 可见→该状态的所有 5 个反事实用全知
    距离，连续；不可见→整个状态不加整形）。
    Visibility continuity (the third pitfall, found 2026-10-15):
    baking "visibility" straight into φ lets an outward step push the
    food out of view, snapping φ back to 0 for a POSITIVE shaping —
    the classic discontinuous-PBRS trap. Fix: the gate is decided at
    the sampled state s (food visible at s -> all 5 counterfactuals
    use omniscient distances, which are continuous; not visible ->
    no shaping at all for this state).
    """
    cfg = env.cfg
    if gates is None:
        gates = decide_gates(env, gate_margin)
    phi = 0.0
    if gates["food"] and env.foods:
        phi -= min(_manhattan(env.agent, f) for f in env.foods)
    if gates["hazard"] and env.hazards:
        phi += min(_manhattan(env.agent, h) for h in env.hazards)
    return float(phi)


def decide_gates(env, gate_margin: float = 0.05) -> Dict[str, bool]:
    """在采样状态判定整形门控。Decide shaping gates at the sampled state.

    food 门开 = 饥饿（energy < setpoint - margin）且视野内有食物
    hazard 门开 = 受伤（integrity < setpoint - margin）且视野内有危险
    """
    cfg = env.cfg
    off = cfg.view // 2
    ar, ac = env.agent

    def _vis(cell: Tuple[int, int]) -> bool:
        return max(abs(cell[0] - ar), abs(cell[1] - ac)) <= off

    food_gate = (env.energy < cfg.energy_setpoint - gate_margin
                 and any(_vis(f) for f in env.foods))
    hazard_gate = (env.integrity < cfg.integrity_setpoint - gate_margin
                   and any(_vis(h) for h in env.hazards))
    return {"food": food_gate, "hazard": hazard_gate}


def rank_actions_at_state(tenv: TextEnv, shape_w: float = 0.05,
                          gate_margin: float = 0.05) -> Dict:
    """在当前状态穷举 5 个动作，返回整形后的排序键。
    Enumerate all 5 actions at the current state; returns the shaped
    ranking keys (one per action).

    进食步跳过整形（第二个坑）：吃到食物后食物在远处重生，φ(s') 用
    "重生后的远食物"会算出大负整形，淹没 +0.33 的真实进食收益。
    Eat steps skip shaping (the second pitfall): after eating, food
    respawns far away and φ(s') computed from the respawned food
    produces a large negative shaping that drowns the +0.33 true
    homeostatic gain.
    """
    snap = {
        "env": copy.deepcopy(tenv.env.__dict__),
        "prev_action": tenv.prev_action,
    }
    gates = decide_gates(tenv.env, gate_margin)
    phi0 = potential(tenv.env, gates, gate_margin)
    keys: List[float] = []
    for a in range(N_ACTIONS):
        tenv.env.__dict__ = copy.deepcopy(snap["env"])
        tenv.prev_action = snap["prev_action"]
        _, r, done, info = tenv.env.step(a)
        if info.get("ate"):
            keys.append(float(r))
        else:
            phi1 = potential(tenv.env, gates, gate_margin)
            keys.append(r + shape_w * (phi1 - phi0))
    tenv.env.__dict__ = copy.deepcopy(snap["env"])
    tenv.prev_action = snap["prev_action"]
    return {"keys": keys}


def _teacher_action(model, env, prev_a: int, device: str) -> int:
    import torch

    v = torch.from_numpy(env.last_obs["vision"]).unsqueeze(0).to(device)
    i = torch.from_numpy(env.last_obs["intero"]).unsqueeze(0).to(device)
    pa = torch.tensor([prev_a], dtype=torch.long, device=device)
    a, _, _ = model.act(v, i, pa, deterministic=True)
    return int(a.item())


class PolicyDriver:
    """用**学生自己**驱动轨迹（L1 自产训练信号的关键）。
    Drive the trajectory with the STUDENT itself (the crux of L1:
    self-generated training signal).

    v3 的偏好对来自教师/随机轨迹——数据分布是"别人走过的地方"。L1 改为
    让学生自己走：它现在错在哪，对比数据就长在哪，偏好对分布随策略
    一起移动（iterated DPO 的本质）。
    v3's pairs come from teacher/random rollouts — the distribution of
    states SOMEONE ELSE visited. L1 lets the student walk: the pairs
    grow wherever it is currently wrong, so the pair distribution moves
    with the policy (that is what iterated DPO means).

    采样而非贪心：贪心会把轨迹压到几条确定性路线上，覆盖反而变窄。
    Sampling, not greedy: greedy collapses trajectories onto a few
    deterministic routes and narrows coverage.
    """

    def __init__(self, model, tok, device: str) -> None:
        from .acthead import action_token_ids
        self.model, self.tok, self.device = model, tok, device
        self.act_ids = action_token_ids(tok)

    def act(self, tenv) -> int:
        from .acthead import act_batch
        from .probe import prompt_ids
        prompts = [prompt_ids(self.tok, tenv.env, tenv.prev_action)]
        acts, _, _ = act_batch(self.model, self.tok, prompts, self.act_ids,
                               self.device, greedy=False)
        return int(acts[0])


def load_policy(policy_dir: Path, device: str = "cpu"):
    """载入学生策略（用于 on-policy 采样）。Load the student policy."""
    from .probe import load_student
    return load_student(Path(policy_dir), device)


def source_shares(pairs: int, policy_frac: float) -> Dict[str, int]:
    """各来源的配额。Per-source quotas.

    有 policy 时：policy 占 policy_frac，其余教师/随机对半。
    With a policy: policy takes policy_frac, the rest splits evenly.
    """
    pol = int(round(pairs * policy_frac)) if policy_frac > 0 else 0
    rest = max(0, pairs - pol)
    return {"teacher": rest // 2 + (rest % 2), "random": rest // 2,
            "policy": pol}


def generate(ckpt_path: Path, out_path: Path, pairs: int = 4000,
             min_gap: float = 0.008, shape_w: float = 0.05,
             gate_margin: float = 0.05, max_steps: int = 400,
             seed: int = 0, device: str = "cpu",
             policy_dir: Optional[Path] = None,
             policy_frac: float = 0.34) -> dict:
    """生成 DPO 偏好对 jsonl。Generate the DPO preference-pair jsonl.

    每行格式（与 SFT 数据同构，chosen/rejected 是 assistant 回复）：
    Each line (same schema as SFT data; chosen/rejected are assistant
    replies):
        {"messages": [...2 条...], "chosen": "right", "rejected": "left",
         "meta": {...}}
    """
    from .gen_data import load_teacher

    model, env_cfg = load_teacher(ckpt_path, device)
    tenv = TextEnv(env_cfg)
    rng = random.Random(seed)
    rows: List[dict] = []
    stats = {"teacher_states": 0, "random_states": 0, "policy_states": 0,
             "dropped_ties": 0}

    shares = source_shares(pairs, policy_frac if policy_dir else 0.0)
    driver = None
    if policy_dir:
        pol_model, pol_tok, pol_dev = load_policy(Path(policy_dir), device)
        driver = PolicyDriver(pol_model, pol_tok, pol_dev)

    def _collect(src: str, quota: int, seed_base: int) -> None:
        """按来源采集状态（三段采集逻辑完全一致，只是驱动不同）。
        Collect states for one source (the three sources differ only in
        their driver)."""
        if quota <= 0:
            return
        key = f"{src}_states"
        stride = max(1, max_steps // 40)  # 每局最多采样 ~40 个状态
        ep = 0
        while stats[key] < quota:
            tenv.reset(seed=seed_base + ep)
            prev_a = NONE_ACTION
            t_idx = 0
            for t in range(max_steps):
                if t_idx % stride == 0 and stats[key] < quota:
                    ks = rank_actions_at_state(tenv, shape_w,
                                               gate_margin)["keys"]
                    c, r_ = int(np.argmax(ks)), int(np.argmin(ks))
                    if ks[c] - ks[r_] >= min_gap:
                        msgs = obs_messages(tenv.env, tenv.prev_action)
                        rows.append({
                            "messages": msgs,
                            "chosen": ACTION_WORDS[c],
                            "rejected": ACTION_WORDS[r_],
                            "meta": {"ep": ep, "t": t, "src": src,
                                     "keys": [round(x, 4) for x in ks]},
                        })
                        stats[key] += 1
                    else:
                        stats["dropped_ties"] += 1
                if src == "teacher":
                    a = _teacher_action(model, tenv.env, prev_a, device)
                elif src == "policy":
                    a = driver.act(tenv)
                else:
                    a = rng.randrange(N_ACTIONS)   # 随机推进 / random walk
                _, _, done, _ = tenv.step(ACTION_WORDS[a])
                prev_a = a
                t_idx += 1
                if done:
                    break
            ep += 1

    _collect("teacher", shares["teacher"], seed * 10_000)
    _collect("random", shares["random"], 1_000_000 + seed * 10_000)
    if driver is not None:
        _collect("policy", shares["policy"], 2_000_000 + seed * 10_000)

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"dpo pairs written: {out_path} ({len(rows)} pairs)")
    print(f"stats: {stats}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="同状态穷举对比 → DPO 偏好对 / same-state exhaustive "
                    "contrast -> DPO preference pairs")
    ap.add_argument("--teacher", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt")
    ap.add_argument("--pairs", type=int, default=4000,
                    help="目标偏好对数量（教师/随机各一半）")
    ap.add_argument("--min-gap", type=float, default=0.008,
                    help="chosen 与 rejected 的最小整形键差（低于丢弃；"
                         "0.008 保留饱腹 rest 对——修复 v2 反向调制的关键）")
    ap.add_argument("--shape-w", type=float, default=0.05,
                    help="势函数整形权重 / potential-shaping weight")
    ap.add_argument("--gate-margin", type=float, default=0.05,
                    help="内感受门控阈值 / interoceptive gate margin")
    ap.add_argument("--policy", type=str, default=None,
                    help="学生适配器目录：用**学生自己**跑一部分轨迹"
                         "（L1 自产训练信号）"
                         " / student adapter dir: let the STUDENT drive "
                         "part of the rollouts (L1 self-generated signal)")
    ap.add_argument("--policy-frac", type=float, default=0.34,
                    help="on-policy 状态的占比（其余教师/随机对半）"
                         " / share of on-policy states")
    ap.add_argument("--out", type=str, default="data/dpo_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(Path(args.teacher), Path(args.out), pairs=args.pairs,
             min_gap=args.min_gap, shape_w=args.shape_w,
             gate_margin=args.gate_margin, seed=args.seed,
             policy_dir=Path(args.policy) if args.policy else None,
             policy_frac=args.policy_frac)


if __name__ == "__main__":
    main()
