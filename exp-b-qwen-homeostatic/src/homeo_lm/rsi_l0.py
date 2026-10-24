"""RSI Level 0：自适应调参（不改权重，只改旋钮）。
RSI Level 0: self-tuning (no weight changes — knobs only).

这是 `docs/RSI-PLAN.md` 里最安全的一层：最坏情况只是浪费算力，训不出
坏策略。全部决策都写进账本，可回滚、可复现。
The safest level in `docs/RSI-PLAN.md`: the worst case is wasted
compute, never a corrupted policy. Every decision lands in the ledger —
rollback-able and reproducible.

四件事 / four things:
1. **旋钮空间**：每个旋钮都有界（越界即结构性拒绝）
   Knob space: every knob is bounded (out of range = structural reject)
2. **不可改清单**：内稳态目标 / 环境物理 / 探针定义 / 门控自身——
   **不看指标直接拒绝**（改了这些，"成绩"就不再可比）
   Never-change list: rejected structurally, regardless of metrics,
   because changing them makes scores incomparable
3. **UCB bandit 提议**：旋钮少、噪声大，先用最笨但最可解释的搜索
   UCB bandit proposal: few knobs, high noise -> use the dumbest
   explainable search first
4. **采纳/回滚**：显著变好才采纳；守卫触发或显著变差 → 回滚到 parent
   Adopt only on significant improvement; guardrail or significant
   regression -> roll back to the parent

本模块**不跑训练、不动显卡**——它只产出"下一步试什么"和"该不该采纳"。
This module never trains and never touches the GPU: it only answers
"what to try next" and "should we adopt it".
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------ 不可改清单
# 这些键出现即结构性拒绝，不看指标。改了它们，"成绩"就不再可比——
# 相当于让考生自己改考卷。
# These keys are rejected structurally, never by metric: changing them
# makes scores incomparable — it is letting the examinee rewrite the
# exam paper.
FORBIDDEN_KEYS = {
    "energy_setpoint",          # 内稳态目标 / homeostatic objective
    "integrity_setpoint",
    "deficit_fn",
    "reward_fn",
    "death_condition",          # 环境物理 / environment physics
    "move_cost",
    "probe_cfg",                # 探针定义 / probe definitions
    "probe_hash",
    "seed_set",
    "gate",                     # 门控自身 / the gate itself
    "forbidden_keys",
    "guardrails",
}


def forbidden_keys_in(change: Dict) -> List[str]:
    """改动里碰到的不可改键。Which never-change keys this change touches."""
    return sorted(k for k in change if k in FORBIDDEN_KEYS)


# -------------------------------------------------------------------- 旋钮空间
class Knob:
    """一个有界旋钮。A bounded knob."""

    def __init__(self, name: str, values: Sequence, stage: str,
                 arg: Optional[str] = None) -> None:
        if not values:
            raise ValueError(f"knob {name} has no values")
        self.name = name
        self.values = list(values)
        self.stage = stage          # 作用于哪个阶段 / which pipeline stage
        self.arg = arg or f"--{name.replace('_', '-')}"

    def contains(self, v) -> bool:
        return v in self.values

    def __repr__(self) -> str:
        return f"Knob({self.name}, {self.values}, stage={self.stage})"


def _frange(lo: float, hi: float, step: float) -> List[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


# L0 允许自动调的旋钮（全部有界；越界即拒绝）
# The knobs L0 may tune automatically (all bounded; out of range = reject)
KNOBS: Dict[str, Knob] = {
    "shape_w": Knob("shape_w", _frange(0.0, 0.15, 0.025), "dpo_pairs",
                    "--shape-w"),
    "gate_margin": Knob("gate_margin", _frange(0.02, 0.15, 0.025),
                        "dpo_pairs", "--gate-margin"),
    "teacher_mix": Knob("teacher_mix", [0.0, 0.25, 0.5, 0.75, 1.0],
                        "dpo_pairs", "--teacher-mix"),
    "entropy_coef": Knob("entropy_coef", _frange(0.0, 0.05, 0.01),
                         "ppo", "--entropy-coef"),
    "curriculum": Knob("curriculum", [0, 1, 2, 3, 4], "env",
                       "--curriculum"),
}


def validate_knobs(change: Dict) -> List[str]:
    """越界/未知旋钮的原因列表（空 = 合法）。
    Reasons why the change is invalid (empty = valid)."""
    bad = [f"forbidden: {k}" for k in forbidden_keys_in(change)]
    for k, v in change.items():
        if k in FORBIDDEN_KEYS:
            continue
        if k not in KNOBS:
            bad.append(f"unknown knob: {k}")
        elif not KNOBS[k].contains(v):
            bad.append(f"out of range: {k}={v} not in {KNOBS[k].values}")
    return bad


def knob_to_args(change: Dict) -> Dict[str, List[str]]:
    """改动 → 各阶段的命令行参数（真正跑训练时用来拼命令）。
    Change -> per-stage CLI args (used to build the real commands)."""
    out: Dict[str, List[str]] = {}
    for k, v in change.items():
        if k in FORBIDDEN_KEYS or k not in KNOBS:
            continue
        knob = KNOBS[k]
        out.setdefault(knob.stage, []).extend([knob.arg, str(v)])
    return out


# ------------------------------------------------------------------ UCB 提议器
def _arm_key(knob: str, value) -> str:
    return f"{knob}={value}"


def ucb_propose(history: Sequence[Dict], knobs: Sequence[str],
                c: float = 1.4, rng_seed: int = 0) -> Dict:
    """UCB1 选出下一个候选改动。UCB1 picks the next candidate change.

    history: 账本条目列表，每条含 `change`（dict）与 `score`（float，越大
    越好）。未试过的臂优先（UCB = +inf）。
    history: ledger entries with `change` (dict) and `score` (float,
    higher is better). Untried arms come first (UCB = +inf).

    为什么不用贝叶斯优化：旋钮少、噪声大、且每次结果都要进账本可复现；
    先用最笨但最可解释的方法，等证明有用再升级。
    Why not Bayesian optimisation: few knobs, high noise, and every
    result must be reproducible from the ledger. Prove it works first.
    """
    import numpy as np

    arms: List[Tuple[str, str, object]] = []   # (arm_key, knob, value)
    for name in knobs:
        if name not in KNOBS:
            continue
        for v in KNOBS[name].values:
            arms.append((_arm_key(name, v), name, v))
    if not arms:
        return {"change": {}, "arm": None, "note": "no knobs"}

    sums: Dict[str, float] = {a[0]: 0.0 for a in arms}
    cnts: Dict[str, int] = {a[0]: 0 for a in arms}
    for e in history:
        score = e.get("score")
        if score is None:
            continue
        for k, v in (e.get("change") or {}).items():
            key = _arm_key(k, v)
            if key in sums:
                sums[key] += float(score)
                cnts[key] += 1
    total = max(1, sum(cnts.values()))

    best, best_ucb = None, -math.inf
    untried = [a for a in arms if cnts[a[0]] == 0]
    if untried:
        # 未试过的臂按确定性顺序依次试（同一 seed 可复现）
        # try untried arms in a deterministic order (reproducible)
        idx = total % len(untried)
        pick = untried[idx]
        return {"change": {pick[1]: pick[2]}, "arm": pick[0],
                "ucb": float("inf"), "untried": True}
    for key, knob, value in arms:
        n, s = cnts[key], sums[key]
        ucb = s / n + c * math.sqrt(math.log(total) / n)
        if ucb > best_ucb:
            best, best_ucb = (knob, value), ucb
    return {"change": {best[0]: best[1]}, "arm": _arm_key(*best),
            "ucb": float(best_ucb), "untried": False}


# ---------------------------------------------------------------- 预算与熔断
class Budget:
    """每周期的试验预算与熔断。Per-cycle trial budget + circuit breaker."""

    def __init__(self, max_trials: int = 5, max_consecutive_fail: int = 3,
                 cool_down_cycles: int = 1) -> None:
        self.max_trials = max_trials
        self.max_consecutive_fail = max_consecutive_fail
        self.cool_down_cycles = cool_down_cycles

    def can_try(self, cycle_trials: int, consecutive_fail: int) -> Tuple[bool,
                                                                         str]:
        if cycle_trials >= self.max_trials:
            return False, f"budget: {cycle_trials}/{self.max_trials} trials"
        if consecutive_fail >= self.max_consecutive_fail:
            return False, (f"circuit breaker: {consecutive_fail} consecutive "
                           f"failures — frozen, needs a human")
        return True, ""


# -------------------------------------------------------------------- 反振荡
def oscillating(history: Sequence[Dict], knob: str,
                window: int = 4) -> bool:
    """某旋钮是否在最近几轮来回翻转（锯齿）。
    Is this knob zig-zagging over the last few entries?

    振荡说明改进过程在噪声里打转，而不是在爬山——必须停下来交给人。
    Oscillation means the loop is spinning in noise instead of climbing;
    stop and hand it to a human.
    """
    vals = [e.get("change", {}).get(knob) for e in history
            if knob in (e.get("change") or {})]
    vals = vals[-window:]
    if len(vals) < 3:
        return False
    flips = sum(1 for i in range(len(vals) - 2)
                if (vals[i] - vals[i + 1]) * (vals[i + 1] - vals[i + 2]) < 0)
    return flips >= 2


# ---------------------------------------------------------------- 采纳与回滚
def decide(before: Sequence[float], after: Sequence[float],
           guardrails_ok: bool = True, n_boot: int = 2000,
           seed: int = 0) -> Dict:
    """采纳 / 拒绝 / 回滚。Adopt / reject / roll back.

    - 守卫失败 → **回滚**（不看指标）
    - 显著变差 → 回滚
    - 显著变好 → 采纳
    - 其余 → 拒绝（不采纳，但也不回滚：没有证据说它更糟）
    Guardrail violated -> ROLL BACK (regardless of metrics); significant
    regression -> roll back; significant improvement -> adopt; anything
    else -> reject (do not adopt, but no rollback either: no evidence it
    is worse).
    """
    from .rsi import significant

    if not guardrails_ok:
        return {"verdict": "rollback", "reason": "guardrail violated"}
    res = significant(before, after, n_boot=n_boot, seed=seed)
    if res["regressed"]:
        return {"verdict": "rollback", "reason": "significant regression",
                "stats": res}
    if res["significant"]:
        return {"verdict": "adopt", "reason": "significant improvement",
                "stats": res}
    return {"verdict": "reject", "reason": "no significant difference",
            "stats": res}


def gate(change: Dict, probe_hash_now: str,
         probe_hash_baseline: Optional[str],
         budget: Optional[Budget] = None, cycle_trials: int = 0,
         consecutive_fail: int = 0) -> Dict:
    """门控：所有硬约束的检查口。The gate: every hard constraint at once.

    返回 (ok, reasons)。不看指标——只看"这个改动有没有资格被考虑"。
    Returns (ok, reasons). It never looks at metrics, only at whether
    the change is even eligible.
    """
    reasons = validate_knobs(change)
    # 探针锁：考试换了，本轮全部作废
    # probe lock: if the exam changed, this round is void
    if probe_hash_baseline is not None and probe_hash_now != probe_hash_baseline:
        reasons.append(f"probe changed: {probe_hash_baseline} -> "
                       f"{probe_hash_now} (all adoptions this round are void)")
    if not change:
        reasons.append("empty change")
    if budget is not None:
        ok_b, why = budget.can_try(cycle_trials, consecutive_fail)
        if not ok_b:
            reasons.append(why)
    return {"ok": not reasons, "reasons": reasons}


def cycle_plan(history: Sequence[Dict], knobs: Sequence[str],
               probe_hash_now: str, probe_hash_baseline: Optional[str],
               budget: Optional[Budget] = None, cycle_trials: int = 0,
               consecutive_fail: int = 0) -> Dict:
    """一个 L0 周期的完整计划（不执行任何训练）。
    A full L0 cycle plan (executes no training)."""
    proposal = ucb_propose(history, knobs)
    g = gate(proposal["change"], probe_hash_now, probe_hash_baseline,
             budget, cycle_trials, consecutive_fail)
    osc = [k for k in proposal["change"] if oscillating(history, k)]
    if osc:
        g["ok"] = False
        g["reasons"].append(f"oscillating knobs: {osc} — stop, needs a human")
    return {"proposal": proposal, "gate": g,
            "args": knob_to_args(proposal["change"]) if g["ok"] else {}}


def main() -> None:  # pragma: no cover - 手动查看用 / manual inspection
    ap = __import__("argparse").ArgumentParser(
        description="RSI L0 计划器（只出计划，不跑训练）"
                    " / RSI L0 planner (plan only, no training)")
    ap.add_argument("--ledger", type=str, default="runs/rsi_ledger.jsonl")
    ap.add_argument("--knobs", type=str, default=",".join(KNOBS))
    args = ap.parse_args()
    from pathlib import Path
    from .rsi import Ledger
    hist = Ledger(Path(args.ledger)).read()
    print(json.dumps(cycle_plan(hist, args.knobs.split(","), "now", None),
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
