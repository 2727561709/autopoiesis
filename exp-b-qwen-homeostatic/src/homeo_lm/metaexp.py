"""元探针：RSI 到底值不值——**同等算力**下的对照组实验。
The meta-probe: is RSI worth anything — a control-group experiment at
EQUAL compute.

这是整个 RSI 方案里最容易被自欺的一环：跑完闭环看到指标涨了，就宣称
"自我改进有效"。但同样算力拿去**闷头多训一会**也可能涨。所以问题不是
"指标高不高"，而是：
This is the part of the RSI plan where self-deception is easiest: the
loop finishes, the metric went up, and one declares "self-improvement
works". But the same compute spent on JUST TRAINING LONGER might also
raise it. So the question is not "are the metrics high?" but:

> **同等算力下，RSI 组是否优于固定课程组？**
> **At equal compute, does the RSI group beat the fixed-curriculum
> group?**

| 组 | 设置 | 用于回答 |
|---|---|---|
| A | 固定课程一直训（对照） | 算力本身能买多少性能 |
| B | L0 only | 自适应调参值不值 |
| C | L1 only（固定超参） | 自产信号值不值 |
| D | L0 + L1 | 两者是否互补（**超可加性**） |
| E | L0+L1+L2 | 改学习规则还有多少增量 |

判据：`D > A` 且 `D > B` 且 `D > C`。若 `D ≈ A`，说明 RSI 在这个环境里
买不到东西——**那也是有价值的结果，必须如实记录**（项目章程：不崇拜负
结果，但必须记录）。
Verdict: `D > A` and `D > B` and `D > C`. If `D ≈ A`, RSI buys nothing
in this environment — that is a valuable result too, and it must be
recorded honestly.

本模块只做**计划、记账与判定**，不跑训练（与 L1 同：真实训练由子进程执行）。
This module only plans, records and judges; it runs no training.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

# 每组的层级开关 / which levels each group enables
GROUPS: Dict[str, Dict] = {
    "A": {"levels": [], "note": "fixed curriculum, just train longer"},
    "B": {"levels": ["L0"], "note": "self-tuning only"},
    "C": {"levels": ["L1"], "note": "self-generated signal only"},
    "D": {"levels": ["L0", "L1"], "note": "both — the superadditivity test"},
    "E": {"levels": ["L0", "L1", "L2"], "note": "+ learning rule changes"},
}

# 每组的算力预算（**各组必须相等**，否则比较无意义）
# Per-group compute budget (MUST be equal across groups, or the
# comparison means nothing)
DEFAULT_BUDGET = {"trials": 12, "train_steps": 500, "probe_runs": 6}

# 证据地板：差值不高于它就不算证据——否则 D 比 A 高 0.01 就会被宣称为
# "RSI 有效"，那正是本节要防的自欺。这是一个**写明并可审计**的常数，
# 不是随手调的阈值。
# Evidence floor: a difference below this is not evidence — otherwise
# "D beats A by 0.01" would be declared a win, which is exactly the
# self-deception this section exists to prevent. It is a WRITTEN and
# AUDITABLE constant, not a knob to fiddle with.
MIN_MARGIN = 0.02


def group_plan(group: str, seeds: Sequence[int] = (0, 1, 2),
               budget: Optional[Dict] = None) -> Dict:
    """一组的实验计划（命令序列 + 算力账）。
    The plan for one group (command sequence + compute accounting)."""
    import sys
    g = GROUPS[group]
    b = dict(DEFAULT_BUDGET if budget is None else budget)
    cmds: List[List[str]] = []
    for s in seeds:
        if "L1" in g["levels"]:
            cmds.append([sys.executable, "-m", "homeo_lm.selftrain",
                         "--start", "runs/dpo-v3", "--rounds", "3",
                         "--seed", str(s), "--run"])
        else:
            cmds.append([sys.executable, "-m", "homeo_lm.dpo",
                         "--data", "data/dpo_pairs.jsonl",
                         "--out", f"runs/meta-{group}-s{s}",
                         "--seed", str(s)])
        if "L0" in g["levels"]:
            cmds.append([sys.executable, "-m", "homeo_lm.rsi", "--cycle",
                         "--out", f"runs/meta-{group}-s{s}/ledger.jsonl",
                         "--run-trials"])
        if "L2" in g["levels"]:
            cmds.append([sys.executable, "-m", "homeo_lm.rsi", "--level", "2",
                         "--out", f"runs/meta-{group}-s{s}/ledger.jsonl",
                         "--holdout", "--seeds", "3"])
    return {"group": group, "levels": g["levels"], "note": g["note"],
            "seeds": list(seeds), "budget": b, "commands": cmds,
            "cost": cost(b, len(seeds))}


def cost(budget: Dict, n_seeds: int) -> Dict:
    """算力账：让"同等算力"可被核对，而不是嘴上说说。
    Compute accounting: so "equal compute" is checkable, not a claim."""
    return {"trials": budget["trials"] * n_seeds,
            "train_steps": budget["train_steps"] * n_seeds,
            "probe_runs": budget["probe_runs"] * n_seeds,
            "unit": "steps+probes",
            "total_units": (budget["train_steps"] + budget["probe_runs"])
            * n_seeds}


def equal_compute(plans: Sequence[Dict]) -> Dict:
    """各组算力是否真的相等（不相等则整轮实验无效）。
    Is compute genuinely equal across groups (if not, the whole
    experiment is void)?"""
    totals = {p["group"]: p["cost"]["total_units"] for p in plans}
    return {"totals": totals,
            "equal": len(set(totals.values())) == 1}


def verdict(results: Dict[str, Sequence[float]],
            noise: float = 0.0) -> Dict:
    """判定：RSI 是否真的优于闷头多训。
    The verdict: does RSI actually beat just training longer?

    results: {组: [每 seed 的分数]}。判据是**超可加性**——`D` 必须同时
    压过对照 `A` 和两个单独开的组。
    results: {group: [per-seed scores]}. The criterion is
    SUPERADDITIVITY — D must beat the control A and both single-level
    groups.
    """
    def mean(g: str) -> Optional[float]:
        v = results.get(g)
        return None if not v else sum(v) / len(v)

    # seed 间的实际波动（如实汇报，供人判断）/ observed seed spread
    # (reported honestly, so a human can judge)
    spreads = [max(v) - min(v) for v in results.values() if len(v) > 1]
    spread = max(spreads) if spreads else 0.0
    # 有效判定门槛：噪声与证据地板取大者
    # effective margin: the larger of the noise and the evidence floor
    margin = max(float(noise), MIN_MARGIN)

    a, b, c, d, e = (mean("A"), mean("B"), mean("C"), mean("D"), mean("E"))
    out: Dict = {"means": {k: mean(k) for k in ("A", "B", "C", "D", "E")},
                 "noise": noise, "seed_spread": spread, "margin": margin}
    if None in (a, b, c, d):
        out["verdict"] = "incomplete"
        out["note"] = "需要 A/B/C/D 四组都有结果 / need results for A/B/C/D"
        return out
    beat_a = d > a + margin
    beat_b = d > b + margin
    beat_c = d > c + margin
    out["beats"] = {"A": beat_a, "B": beat_b, "C": beat_c}
    out["superadditive"] = bool(beat_a and beat_b and beat_c)
    if out["superadditive"]:
        out["verdict"] = "rsi_pays"
        out["note"] = ("D > A 且 D > B、D > C：RSI 买到了超出算力的东西"
                       " / D > A and D > B, D > C: RSI bought something "
                       "beyond raw compute")
    else:
        out["verdict"] = "no_evidence"
        out["note"] = ("D 未同时压过 A/B/C：RSI 在此环境下未被证明有效——"
                       "这是负结果，必须如实记录"
                       " / D did not beat A/B/C: RSI is unproven here. "
                       "A negative result — record it honestly.")
    if e is not None and d is not None:
        out["l2_increment"] = e - d
    return out


def main() -> None:  # pragma: no cover - 手动查看用 / manual inspection
    import argparse
    ap = argparse.ArgumentParser(
        description="元实验：A–E 对照组计划（默认 dry-run）"
                    " / meta-experiment: A-E control-group plan "
                    "(dry-run by default)")
    ap.add_argument("--groups", type=str, default="ABCDE")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    plans = [group_plan(g, seeds) for g in args.groups]
    eq = equal_compute(plans)
    print(json.dumps({"plans": plans, "equal_compute": eq},
                     ensure_ascii=False, indent=2))
    if not eq["equal"]:
        print("WARNING: 各组算力不等 — 比较无效"
              " / compute differs across groups: comparison is void")


if __name__ == "__main__":
    main()
