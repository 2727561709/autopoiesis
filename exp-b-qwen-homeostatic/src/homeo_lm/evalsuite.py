"""评测 Harness：把四项探针做成**可配置的算例套件**。
Eval harness: the four probes as a CONFIGURABLE SUITE of cases.

为什么需要它：探针散在 `probe.py` 里，每次想换一组参数（不同种子、不同
样本量、不同门槛）就得改命令行；更糟的是**判定口径会漂**——这次说 gap>0.5
算过，下次只看"是不是正的"。
Why it exists: the probes live inside `probe.py` and every change of
parameters (seeds, sample size, thresholds) meant new command lines;
worse, the VERDICT CRITERION DRIFTS — one run says gap > 0.5 passes, the
next only asks whether it is positive.

一个算例 = 一次可复现的评测（名字 + 探针 + 种子 + 样本量 + 门槛）：
A case is one reproducible evaluation (name + probe + seed + samples +
thresholds)::

    Case(name="modulation-main", probe="modulation", seed=50_000,
         samples=3, thresholds={"modulation_gap": (">=", 0.5)})

套件跑完给出：**基线 vs 候选的对比表** + **逐算例的门槛判定**。
The suite returns a BASE vs CANDIDATE comparison table plus a per-case
threshold verdict.

**环境打分，模型永不自评**（约束 #3）：所有指标来自环境侧信号。
The ENVIRONMENT grades, the model never grades itself (constraint #3):

用法 / usage:
    py -3.12 -m homeo_lm.evalsuite --base runs/sft-lora-v2 \
        --cand runs/dpo-v3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .rsi import DEFAULT_PROBE_CFG, probe_hash
from .selftrain import composite_score, metrics_from_report

# 算例可用的探针（与 probe.py 的四个探针一一对应）
# probes a case may reference (mirrors probe.py)
PROBES = ("agreement", "survival", "modulation", "modulation-sweep")

# 指标方向：True = 越大越好 / True = higher is better
HIGHER_IS_BETTER = {
    "agreement": True,
    "survival_rate": True,
    "mean_steps": True,
    "modulation_gap": True,
    "sweep_slope": True,
    "illegal_rate": False,
}

OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
}


@dataclass
class Case:
    """一个可复现的评测算例。One reproducible eval case."""

    name: str
    probe: str = "modulation"
    seed: int = 50_000
    samples: int = 3
    thresholds: Dict[str, Tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probe not in PROBES:
            raise ValueError(f"unknown probe {self.probe!r}; "
                             f"choose from {PROBES}")

    def as_dict(self) -> Dict:
        return {"name": self.name, "probe": self.probe, "seed": self.seed,
                "samples": self.samples,
                "thresholds": {k: list(v) for k, v in
                               self.thresholds.items()}}


def default_suite() -> List[Case]:
    """默认套件：三个主探针 + 剂量-反应曲线。
    The default suite: the three main probes plus the dose-response
    curve.

    门槛取自 v3 的判读口径：调制必须**转正且过 0.5**，非法率不得超 20%。
    Thresholds follow the v3 verdict: modulation must flip POSITIVE and
    clear 0.5; the illegal rate may not exceed 20%.
    """
    return [
        Case(name="modulation-main", probe="modulation", seed=50_000,
             samples=3,
             thresholds={"modulation_gap": (">=", 0.5)}),
        Case(name="survival-main", probe="survival", seed=9_000,
             samples=3,
             thresholds={"illegal_rate": ("<=", 0.20)}),
        Case(name="agreement-main", probe="agreement", seed=100,
             samples=3,
             thresholds={"illegal_rate": ("<=", 0.20)}),
        Case(name="modulation-sweep", probe="modulation-sweep",
             seed=50_000, samples=2,
             thresholds={"sweep_slope": (">", 0.0)}),
    ]


def load_suite(path: Optional[str] = None) -> List[Case]:
    """从 JSON 文件读套件（缺省用 `default_suite()`）。
    Load a suite from JSON (falls back to `default_suite()`)."""
    if not path:
        return default_suite()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("cases", [])
    out = []
    for r in rows:
        th = {k: tuple(v) for k, v in (r.get("thresholds") or {}).items()}
        out.append(Case(name=r["name"], probe=r.get("probe", "modulation"),
                        seed=int(r.get("seed", 50_000)),
                        samples=int(r.get("samples", 3)), thresholds=th))
    return out


def check_thresholds(metrics: Dict, thresholds: Dict) -> Dict:
    """逐条判门槛。Check every threshold.

    缺失指标 = **不知道** = 判不过（不许默认通过，那等于把"没测"当成
    "测过了没问题"）。
    A missing metric is UNKNOWN and FAILS — never a default pass, which
    would treat "not measured" as "measured and fine".
    """
    reasons: List[str] = []
    for key, spec in thresholds.items():
        op, want = spec[0], float(spec[1])
        val = metrics.get(key)
        if val is None or op not in OPS:
            reasons.append(f"{key}: unknown/missing (val={val}, op={op})")
            continue
        if not OPS[op](float(val), want):
            reasons.append(f"{key} {float(val):.3f} fails {op} {want}")
    return {"ok": not reasons, "reasons": reasons}


def compare(base: Dict, cand: Dict, thresholds: Optional[Dict] = None) -> Dict:
    """基线 vs 候选：逐指标比较 + 门槛判定。
    Base vs candidate: per-metric comparison plus the threshold verdict."""
    bm = metrics_from_report(base)
    cm = metrics_from_report(cand)
    th = thresholds or {}
    deltas = {}
    for k in set(bm) | set(cm):
        b, c = bm.get(k), cm.get(k)
        if b is None or c is None:
            deltas[k] = None          # 不知道 / unknown, not 0
        else:
            fb, fc = float(b), float(c)
            deltas[k] = fc - fb if HIGHER_IS_BETTER.get(k, True) \
                else fb - fc          # 正向化 / oriented so > 0 = better
    return {
        "base": bm,
        "candidate": cm,
        "delta": deltas,              # > 0 表示变好 / > 0 means better
        "composite": {"base": composite_score(bm),
                      "candidate": composite_score(cm)},
        "thresholds": check_thresholds(cm, th),
    }


def evaluate(results: Dict[str, Dict], suite: Sequence[Case],
             base_key: str = "base") -> Dict:
    """在多个 checkpoint 结果上跑一遍套件，给出逐算例判定。
    Run the suite over several checkpoint results, one verdict per case."""
    rows = []
    for c in suite:
        base = results.get(base_key) or {}
        row = {"case": c.name, "probe": c.probe,
               "thresholds": {k: list(v) for k, v in c.thresholds.items()},
               "per_ckpt": {}}
        for key, rep in results.items():
            cmp_ = compare(base, rep, c.thresholds) if key != base_key \
                else compare(rep, rep, c.thresholds)
            row["per_ckpt"][key] = {
                "metrics": cmp_["candidate"],
                "delta": cmp_["delta"],
                "pass": cmp_["thresholds"]["ok"],
            }
        rows.append(row)
    overall = all(r["per_ckpt"][k]["pass"]
                  for r in rows for k in r["per_ckpt"]
                  if k != base_key)
    return {"suite": [c.as_dict() for c in suite], "rows": rows,
            "probe_hash": probe_hash(DEFAULT_PROBE_CFG),
            "overall_pass": bool(overall)}


def main() -> None:  # pragma: no cover - CLI 入口 / CLI entry
    ap = argparse.ArgumentParser(
        description="评测套件：基线 vs 候选的对比表 + 门槛判定"
                    " / eval suite: base vs candidate table + thresholds")
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--cand", type=str, default=None)
    ap.add_argument("--suite", type=str, default=None,
                    help="算例套件 JSON 文件 / suite JSON file")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from .selftrain import read_probe_report

    suite = load_suite(args.suite)
    results = {"base": read_probe_report(Path(args.base))}
    if args.cand:
        results["candidate"] = read_probe_report(Path(args.cand))
    out = evaluate(results, suite)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False,
                                             indent=2, default=str),
                                  encoding="utf-8")
        print(f"saved: {args.out}")
