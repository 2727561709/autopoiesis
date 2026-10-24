"""RSI Phase 4：并入 Stage C——训练与部署合一，在线终身改进。
RSI Phase 4: merge into Stage C — training and deployment become one,
online lifelong improvement.

前面三层都在"离线改自己"；P4 把这条闭环接到**正在运行**的策略上。合一
之后最大的新风险是：一次坏更新会**直接换掉线上策略**（离线时最坏只是浪
费算力）。所以这一层的承重件是金丝雀晋升与回归套件。
The first three levels improve the system offline; P4 attaches that loop
to a LIVE policy. The new risk unification brings is that one bad update
would REPLACE the live policy (offline, the worst case was wasted
compute). So the load-bearing parts here are canary promotion and the
regression suite.

四条硬规矩 / four hard rules:
1. **线上策略永不被原地替换**：候选先在影子侧跑，通过金丝雀才晋升
   the live policy is never replaced in place: the candidate runs in
   shadow and is promoted only if the canary passes
2. **任一受监控指标回退即不晋升**（不许"总分高了但某项塌了"）
   ANY monitored metric regressing blocks promotion — no trading one
   metric against the whole
3. **留出集增量必须为正**：线上涨、留出不动 = 只是在背现在的题
   the holdout delta must be positive: live up but holdout flat means
   it is memorising the current stream
4. **长期健康**：>10 周期不许退化、不许振荡、长期不破最好成绩要报警
   long-run health: no degradation, no oscillation over >10 cycles, and
   a plateau alarm

用法 / usage（默认 dry-run）:
    py -3.12 -m homeo_lm.lifelong --ledger runs/rsi_ledger.jsonl
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# 受监控的指标（全部"越大越好"；非法率是负权重，单独处理）
# Monitored metrics (all higher-is-better; illegal rate is handled with
# a negative sign separately)
MONITORED = ("modulation_gap", "survival_rate", "agreement")

PROMOTION = {
    "min_holdout_delta": 0.02,   # 留出集至少要涨这么多 / minimum holdout gain
    "regression_tol": 0.0,       # 任一指标不许回退 / no metric may regress
    "illegal_max": 0.20,         # 非法率上限 / illegal-rate ceiling
}


@dataclass
class Cycle:
    """一个在线周期的记录。One online cycle record."""

    idx: int
    incumbent: str
    candidate: Optional[str] = None
    verdict: str = "pending"
    metrics: Dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%S"))


# ------------------------------------------------------------------ 晋升判定
def promote(candidate: Dict, incumbent: Dict, holdout: Optional[Dict] = None,
            rules: Optional[Dict] = None) -> Dict:
    """金丝雀判定：候选能不能顶替线上策略。
    Canary verdict: may the candidate replace the live policy?

    条件（**全部**满足）/ conditions (ALL must hold):
    - 每个受监控指标 ≥ 线上 − 容差（不许用一项换另一项）
      every monitored metric >= live - tol (no trading one for another)
    - 留出集增量 > 阈值（防止只是背当前流）
      holdout delta above the threshold (or it is memorising the stream)
    - 非法率不超上限
      illegal rate under the ceiling
    """
    r = dict(PROMOTION if rules is None else rules)
    reasons: List[str] = []
    for k in MONITORED:
        c, i = candidate.get(k), incumbent.get(k)
        if c is None or i is None:
            reasons.append(f"{k}: 数据缺失，不许晋升 / missing, no promotion")
            continue
        if float(c) < float(i) - r["regression_tol"]:
            reasons.append(f"{k} 回退 / regressed: {c} < {i}")
    if float(candidate.get("illegal_rate", 0.0)) > r["illegal_max"]:
        reasons.append(f"illegal_rate {candidate.get('illegal_rate')} > "
                       f"{r['illegal_max']}")
    delta = None
    if holdout:
        ch, ih = holdout.get("candidate"), holdout.get("incumbent")
        if ch is None or ih is None:
            reasons.append("holdout 数据缺失 / holdout missing")
        else:
            delta = float(ch) - float(ih)
            if delta < r["min_holdout_delta"]:
                reasons.append(f"holdout delta {delta:+.3f} < "
                               f"{r['min_holdout_delta']}")
    else:
        reasons.append("holdout 未测：不许晋升（规则 3）"
                       " / holdout not measured: no promotion (rule 3)")
    return {"ok": not reasons, "reasons": reasons, "holdout_delta": delta}


# -------------------------------------------------------------------- 漂移
def drift(live: Dict, reference: Dict, tol: float = 0.10) -> List[str]:
    """线上行为相对基线漂移了多少（用于触发改进周期）。
    How far live behaviour has drifted from the reference (used to
    trigger an improvement cycle)."""
    out = []
    for k in MONITORED:
        a, b = live.get(k), reference.get(k)
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) > tol:
            out.append(f"{k}: {a} vs {b} (Δ {float(a) - float(b):+.3f})")
    return out


# ---------------------------------------------------------------- 节流阀
def throttle(steps_since_last: int, seconds_since_last: float,
             min_steps: int = 5_000, min_seconds: float = 600.0,
             drifted: bool = False) -> Dict:
    """该不该开新周期：改进不能霸占线上算力。
    Should a new cycle start: improvement must not hog the live budget.

    漂移是例外——行为明显变了可以提前开，但不能绕过时间下限太多。
    Drift is an exception: visible behaviour change may start a cycle
    early, but never below the hard floor.
    """
    if steps_since_last < min_steps and not drifted:
        return {"run": False, "reason": f"steps {steps_since_last} < "
                                        f"{min_steps} and no drift"}
    if seconds_since_last < min_seconds * (0.5 if drifted else 1.0):
        return {"run": False, "reason": f"only {seconds_since_last:.0f}s "
                                        f"since the last cycle"}
    return {"run": True, "reason": "drift" if drifted else "scheduled"}


# ------------------------------------------------------------ 长期健康监控
def health(rows: Sequence[Dict], window: int = 10,
           plateau_k: int = 5) -> Dict:
    """P4 的验收口径：长期无退化、无振荡、长期不破纪录要报警。
    P4's acceptance criterion: no degradation, no oscillation over a long
    run, and a plateau alarm.

    - **退化**：窗口内的最好成绩低于窗口之前的最好成绩
      degraded: the best inside the window is worse than before it
    - **平台**：连续 K 个周期没有刷新最好成绩（可能是停滞，也可能只是
      到顶了——两种情况都要人来看）
      plateau: no new best for K cycles (either stuck or actually at the
      ceiling — both need a human)
    - **振荡**：任一旋钮在窗口内来回翻转（复用 L0 的检测）
      oscillating: any knob zig-zagging within the window
    """
    from .rsi_l0 import oscillating

    scores = [r.get("score") for r in rows if r.get("score") is not None]
    window_rows = list(rows)[-window:]
    best_ever = max(scores) if scores else None
    in_window = [r.get("score") for r in window_rows
                 if r.get("score") is not None]
    best_window = max(in_window) if in_window else None
    prior = list(scores)[:-window] if len(scores) > window else []
    best_prior = max(prior) if prior else None

    degraded = bool(best_prior is not None and best_window is not None
                    and best_window < best_prior)
    plateau = False
    if len(scores) >= plateau_k:
        last_k = list(scores)[-plateau_k:]
        plateau = bool(best_ever is not None
                       and max(last_k) < best_ever)
    knobs: List[str] = []
    for r in window_rows:
        for k in (r.get("change") or {}):
            if k not in knobs:
                knobs.append(k)
    osc = [k for k in knobs if oscillating(window_rows, k, window=window)]
    return {"cycles": len(rows), "window": window,
            "best_ever": best_ever, "best_in_window": best_window,
            "degraded": degraded, "plateau": plateau,
            "oscillating_knobs": osc,
            "ok": not (degraded or osc)}


# ------------------------------------------------------------ 在线循环骨架
class LifelongLoop:
    """训练/部署合一的驱动骨架（**不自动执行训练**）。
    The skeleton that unifies training and deployment (it does NOT run
    training by itself).

    每一步都返回"建议做什么"，真正的训练/探针仍由 `selftrain` /
    `probe` 以子进程执行（同 L1 的理由：不在一个进程里叠模型）。
    Every step returns a recommendation; real training and probes still
    run through `selftrain` / `probe` as subprocesses (same reason as L1:
    never stack models in one process)."""

    def __init__(self, ledger_path: str = "runs/rsi_ledger.jsonl",
                 rules: Optional[Dict] = None) -> None:
        from .rsi import Ledger
        self.ledger = Ledger(ledger_path)
        self.rules = dict(PROMOTION if rules is None else rules)

    def step(self, live_metrics: Dict, reference: Dict,
             candidate: Optional[Dict] = None,
             candidate_metrics: Optional[Dict] = None,
             holdout: Optional[Dict] = None,
             steps_since_last: int = 0,
             seconds_since_last: float = 0.0) -> Dict:
        """一次心跳：要不要开周期、候选能不能晋升、整体健康吗。
        One heartbeat: start a cycle? promote the candidate? healthy?"""
        drifts = drift(live_metrics, reference)
        th = throttle(steps_since_last, seconds_since_last,
                      drifted=bool(drifts))
        promo = None
        if candidate and candidate_metrics is not None:
            promo = promote(candidate_metrics, live_metrics, holdout,
                            self.rules)
        h = health(self.ledger.read())
        out = {"throttle": th, "drift": drifts, "promotion": promo,
               "health": h}
        if promo is not None:
            verdict = "adopt" if promo["ok"] else "reject"
            self.ledger.append({
                "tag": "lifelong", "verdict": verdict,
                "change": {"level": "P4", "candidate": str(candidate)},
                "score": (candidate_metrics or {}).get("modulation_gap"),
                "metrics": candidate_metrics or {},
                "reasons": promo["reasons"],
            })
        return out


def main() -> None:  # pragma: no cover - 手动查看用 / manual inspection
    import argparse
    ap = argparse.ArgumentParser(
        description="P4 在线循环（只读账本，默认不执行任何训练）"
                    " / P4 online loop (reads the ledger; runs no "
                    "training by default)")
    ap.add_argument("--ledger", type=str, default="runs/rsi_ledger.jsonl")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    from .rsi import Ledger
    rows = Ledger(args.ledger).read()
    print(json.dumps(health(rows, window=args.window),
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
