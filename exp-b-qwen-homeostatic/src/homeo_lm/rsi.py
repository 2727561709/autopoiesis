"""RSI 控制器骨架（Phase 0–1）：测量 → 记账。**暂不自动改任何东西**。
RSI controller skeleton (Phase 0-1): measure -> record. It does NOT
modify anything by itself yet.

这是 `docs/RSI-PLAN.md` 里 P0/P1 的最小第一步（MVI）：先把**尺子和账本**
做出来，之后每一次"自我改进"才有东西可对比。在此之前自动修改等于没有
尺子就动刀。
This is the MVI of P0/P1 in `docs/RSI-PLAN.md`: build the ruler and the
ledger first, so that every later self-modification has something to be
compared against. Automatic modification before that is surgery without
a ruler.

三件事 / three things:
1. **种子三分** train / select / holdout：留出集冻结，系统不能对着同一套
   题反复练（v2 反向调制正是背答案的后果）。
   Three-way seed split; the holdout is frozen so the system cannot
   drill on its own exam (v2's inversion was exactly that).
2. **探针锁**：把探针配置哈希写进账本；哈希变了，本轮所有采纳作废。
   Probe lock: the probe config hash goes into the ledger; if it
   changes, every adoption in this round is void.
3. **显著性判定**：Δ > 2σ（bootstrap CI），而非"看起来涨了"。
   Significance: Δ > 2σ (bootstrap CI), not "looks higher".

用法 / usage:
    py -3.12 -m homeo_lm.rsi --ckpt runs/dpo-v3 --tag baseline-A
    py -3.12 -m homeo_lm.rsi --ckpt runs/dpo-v3 --tag candidate-1 \
        --parent <parent-id> --change '{"shape_w": 0.08}'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------ 种子三分
SEED_OFFSET = {"train": 0, "select": 500_000, "holdout": 1_000_000}


def seed_set(name: str, n: int, base: int = 7_000) -> List[int]:
    """某一用途的种子集合（互不重叠、确定性）。
    The seeds for one purpose (disjoint across purposes, deterministic).
    """
    if name not in SEED_OFFSET:
        raise ValueError(f"unknown seed set: {name}")
    off = SEED_OFFSET[name]
    return [off + base + i for i in range(n)]


def seeds_are_disjoint(*sets: Sequence[int]) -> bool:
    """留出集与训练/选择集必须无交集。Holdout must not overlap."""
    seen: set = set()
    for s in sets:
        if seen & set(s):
            return False
        seen |= set(s)
    return True


# -------------------------------------------------------------------- 探针锁
def probe_hash(cfg: Dict) -> str:
    """探针配置的稳定哈希（键排序后序列化）。
    Stable hash of the probe config (keys sorted before serialising).

    探针是"考试"，考试变了成绩就不可比——所以配置本身必须进账本。
    The probe is the exam; if the exam changes, scores are not
    comparable, so the config itself must be recorded.
    """
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


DEFAULT_PROBE_CFG: Dict = {
    "agreement_steps": 200,
    "agreement_trajs": 4,
    "survival_episodes": 20,
    "modulation_trials": 40,
    "sweep_trials": 20,
    "sweep_energies": [0.2, 0.4, 0.6, 0.8, 1.0],
    "batch_size": 32,
    "version": 1,
}


# -------------------------------------------------------------- 显著性判定
def bootstrap_ci(xs: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """均值的 bootstrap 置信区间。Bootstrap CI of the mean."""
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def significant(before: Sequence[float], after: Sequence[float],
                n_boot: int = 2000, seed: int = 0) -> Dict:
    """判断改动是否真的有效：Δ 的 95% CI 是否整体 > 0。
    Did the change really help: is the 95% CI of Δ entirely > 0?

    只看"均值涨了"会被噪声骗——v1/v2 之间 0.49→0.59 的那种"进步"在
    单 seed 上根本分不出真假。
    "The mean went up" is noise-prone; 0.49 -> 0.59 between v1 and v2
    is not distinguishable from noise on a single seed.
    """
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")),
                "significant": False, "note": "empty sample"}
    boot = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        boot.append(rng.choice(b, b.size, replace=True).mean()
                    - rng.choice(a, a.size, replace=True).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(b.mean() - a.mean()), "ci": (float(lo), float(hi)),
            "significant": bool(lo > 0),
            "regressed": bool(hi < 0),
            "n_before": int(a.size), "n_after": int(b.size)}


# ------------------------------------------------------------------ 演进账本
class Ledger:
    """演进账本：每次测量/改动一行 jsonl，可回溯、可回滚。
    The evolution ledger: one jsonl line per measurement / change —
    traceable and rollback-able."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: Dict) -> Dict:
        entry = dict(entry)
        entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        entry.setdefault("id", hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str)
            .encode("utf-8")).hexdigest()[:12])
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def read(self) -> List[Dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def last(self) -> Optional[Dict]:
        rows = self.read()
        return rows[-1] if rows else None


# ------------------------------------------------------------- 守卫（只报告）
GUARDRAILS = {"collapse_top_max": 0.60, "illegal_rate_max": 0.20}


def check_guardrails(report: Dict) -> Dict:
    """守卫检查：这一版只报告，不自动处置（P1 才接自动回滚）。
    Guardrail check: this version only REPORTS (auto-rollback comes in
    P1). 熵地板/坍缩熔断是 v1 教训的直接编码。
    The collapse breaker encodes the v1 lesson directly."""
    out = {"ok": True, "violations": []}
    surv = report.get("survival", {})
    agree = report.get("agreement", {})
    illegal = max(agree.get("illegal_rate", 0.0),
                  surv.get("illegal_rate", 0.0))
    if illegal > GUARDRAILS["illegal_rate_max"]:
        out["ok"] = False
        out["violations"].append(f"illegal_rate {illegal:.3f} > "
                                 f"{GUARDRAILS['illegal_rate_max']}")
    mod = report.get("modulation", {})
    if mod and mod.get("modulation_gap", 0.0) < 0:
        out["violations"].append(
            "modulation_gap < 0 — inverted modulation (v2 failure mode)")
    return out


def run_report(ckpt: Path, teacher: Path, device: Optional[str] = None,
               probe_cfg: Optional[Dict] = None,
               batch_size: int = 32) -> Dict:
    """跑一次完整测量（不写文件、不改任何东西）。
    Run one full measurement (writes nothing, changes nothing)."""
    from .gen_data import load_teacher
    from .probe import (agreement_probe, load_student, modulation_probe,
                        modulation_sweep, survival_probe)
    cfg = dict(probe_cfg or DEFAULT_PROBE_CFG)
    teacher_model, env_cfg = load_teacher(Path(teacher), "cpu")
    student, tok, device = load_student(Path(ckpt),
                                        device or ("cuda" if _cuda() else "cpu"))
    bs = batch_size or cfg.get("batch_size", 32)
    return {
        "checkpoint": str(ckpt),
        "device": device,
        "probe_cfg": cfg,
        "probe_hash": probe_hash(cfg),
        "agreement": agreement_probe(
            student, tok, teacher_model, env_cfg, device=device,
            steps=cfg["agreement_steps"], trajs=cfg["agreement_trajs"],
            batch_size=bs),
        "survival": survival_probe(
            student, tok, env_cfg, device=device,
            episodes=cfg["survival_episodes"], batch_size=bs),
        "modulation": modulation_probe(
            student, tok, env_cfg, device=device,
            trials=cfg["modulation_trials"], batch_size=bs),
        "modulation-sweep": modulation_sweep(
            student, tok, env_cfg, device=device,
            trials=cfg["sweep_trials"], batch_size=bs),
    }


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run_l2(args) -> None:
    """L2：诊断 → 假设 → 门控（仍不自动执行训练）。
    L2: diagnose -> hypothesize -> gate (still executes no training).

    指标未提供时，从账本最后一条记录里取（"上一轮测出来的数"）。
    If no metrics are given, take them from the last ledger entry.
    """
    from .metacog import diagnose, l2_gate, rank

    ledger = Ledger(Path(args.out))
    rows = ledger.read()
    if args.metrics_file:
        metrics = json.loads(Path(args.metrics_file).read_text(
            encoding="utf-8"))
    elif args.metrics:
        metrics = json.loads(args.metrics)
    else:
        metrics = (rows[-1].get("metrics") if rows else None) or {}
    hyps = diagnose(metrics)
    ranked = rank(hyps, rows)
    print(json.dumps({"metrics": metrics, "hypotheses": ranked},
                     ensure_ascii=False, indent=2, default=str))
    if not hyps:
        print("no hypothesis fired — nothing to change (this is the "
              "correct outcome when metrics are healthy)")
        return
    top = hyps[0]
    change = {top.knob: top.values[len(top.values) // 2]}
    g = l2_gate(top, holdout_checked=args.holdout, seeds=args.seeds,
                change=change)
    print(json.dumps({"hypothesis": top.as_dict(), "proposed_change": change,
                      "gate": g}, ensure_ascii=False, indent=2,
                     default=str))
    if not g["ok"]:
        print("L2 GATE REJECTED:", "; ".join(g["reasons"]))
        return
    print("L2 DRY-RUN: 只给出假设与变更，未执行任何训练"
          " / hypothesis and change only; no training executed")
    ledger.append({"tag": f"L2-{top.id}", "verdict": "plan",
                   "change": {"hypothesis": top.id, **change},
                   "probe_hash": probe_hash(DEFAULT_PROBE_CFG),
                   "prediction": top.prediction})


def _run_cycle(args) -> None:
    """一个 L0 周期：提议 → 门控 →（可选）试验 → 判定 → 记账。
    One L0 cycle: propose -> gate -> (optional) trial -> decide -> record.

    默认 **dry-run**：只打印"下一步该试什么"和对应的训练命令，不执行。
    因为一次试验要真跑训练 + 全套探针，在内存紧张的机器上必须先由人来
    确认时机。
    Default is DRY-RUN: it prints what to try next and the matching
    training commands, and executes nothing. A trial means real training
    plus the full probe suite, so a human must pick the moment on a
    memory-tight machine.
    """
    from .rsi_l0 import KNOBS as L0_KNOBS
    from .rsi_l0 import Budget, cycle_plan

    knobs = args.knobs.split(",") if args.knobs else list(L0_KNOBS)
    ledger = Ledger(Path(args.out))
    hist = ledger.read()
    cfg = dict(DEFAULT_PROBE_CFG)
    cfg["batch_size"] = args.batch_size
    ph = probe_hash(cfg)
    base = next((e.get("probe_hash") for e in hist if e.get("verdict") == "baseline"),
                None)
    budget = Budget(max_trials=args.max_trials)
    fails = 0
    for e in reversed(hist):
        if e.get("verdict") in ("adopt", "reject"):
            fails = 0 if e.get("verdict") == "adopt" else fails + 1
        if e.get("verdict") == "adopt":
            break
    plan = cycle_plan(hist, knobs, ph, base, budget,
                      cycle_trials=0, consecutive_fail=fails)
    print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
    if not plan["gate"]["ok"]:
        print("GATE REJECTED:", "; ".join(plan["gate"]["reasons"]))
        return
    cmds = []
    for stage, extra in plan["args"].items():
        if stage == "dpo_pairs":
            cmds.append("py -3.12 -m homeo_lm.gen_dpo_data " + " ".join(extra))
        elif stage == "ppo":
            cmds.append("py -3.12 -m homeo_lm.ppo " + " ".join(extra))
        elif stage == "env":
            cmds.append(f"[env stage] {' '.join(extra)}")
    for c in cmds:
        print("CMD:", c)
    if not args.run_trials:
        print("DRY-RUN: 未执行任何训练（加 --run-trials 才真跑）"
              " / nothing executed (add --run-trials to actually run)")
        ledger.append({"tag": f"plan-{len(hist)}", "verdict": "plan",
                       "change": plan["proposal"]["change"],
                       "probe_hash": ph, "commands": cmds})
        return
    # 真跑：训练 → 测量 → 判定 → 记账
    # real run: train -> measure -> decide -> record
    raise SystemExit("trial execution is not wired yet (P1 hardening)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RSI 控制器（P0/P1）：测量 → 记账 →（L0）自动调参"
                    " / RSI controller: measure -> record -> (L0) tuning")
    ap.add_argument("--ckpt", type=str, default="runs/dpo-v3")
    ap.add_argument("--teacher", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt")
    ap.add_argument("--out", type=str, default="runs/rsi_ledger.jsonl")
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--parent", type=str, default=None,
                    help="上一版账本条目 id / previous ledger entry id")
    ap.add_argument("--change", type=str, default="{}",
                    help='本次改动（JSON），基线留空 / the change (JSON), '
                         'empty for a baseline')
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cycle", action="store_true",
                    help="跑一个 L0 周期：提议 → 门控 →（可选）试验 → 判定"
                         " / run one L0 cycle: propose -> gate -> "
                         "(optional) trial -> decide")
    ap.add_argument("--run-trials", action="store_true",
                    help="真正执行训练（默认 dry-run：只打印命令）"
                         " / actually run training (default: dry-run, "
                         "print the commands only)")
    ap.add_argument("--knobs", type=str, default=None,
                    help="允许自动调的旋钮（默认全部 L0 旋钮）"
                         " / knobs allowed for auto-tuning "
                         "(default: all L0 knobs)")
    ap.add_argument("--max-trials", type=int, default=5,
                    help="每周期试验上限 / trials per cycle")
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2],
                    help="自我改进层级：0 调旋钮 / 1 调数据 / 2 调学习规则"
                         " / improvement level: 0 knobs, 1 data, "
                         "2 learning rule")
    ap.add_argument("--metrics", type=str, default=None,
                    help='L2 诊断用指标 JSON / metrics JSON for L2 '
                         'diagnosis')
    ap.add_argument("--metrics-file", type=str, default=None,
                    help="从文件读指标 JSON（Windows 下命令行转义引号太痛"
                         "苦）/ read metrics JSON from a file (quoting "
                         "JSON on the Windows shell is painful)")
    ap.add_argument("--seeds", type=int, default=0,
                    help="L2 门控：已验证的独立 seed 数 / seeds verified")
    ap.add_argument("--holdout", action="store_true",
                    help="L2 门控：留出集已验过 / holdout verified")
    args = ap.parse_args()

    if args.level == 2:
        _run_l2(args)
        return
    if args.cycle:
        _run_cycle(args)
        return

    report = run_report(Path(args.ckpt), Path(args.teacher), args.device,
                        batch_size=args.batch_size)
    guards = check_guardrails(report)
    change = json.loads(args.change) if args.change else {}
    ledger = Ledger(Path(args.out))
    entry = ledger.append({
        "tag": args.tag,
        "parent": args.parent,
        "change": change,
        "ckpt": str(args.ckpt),
        "probe_hash": report["probe_hash"],
        "metrics": {
            "agreement": report["agreement"].get("agreement"),
            "survival": report["survival"].get("survival_rate"),
            "mean_steps": report["survival"].get("mean_steps"),
            "modulation_gap": report["modulation"].get("modulation_gap"),
            "sweep_slope": report["modulation-sweep"].get("slope"),
        },
        "guardrails": guards,
        "verdict": "baseline" if not change else "candidate",
    })
    print(json.dumps({"entry": entry, "metrics": entry["metrics"],
                      "guardrails": guards}, ensure_ascii=False, indent=2))
    print(f"ledger: {ledger.path} ({len(ledger.read())} entries)")
    if not guards["ok"]:
        print("GUARDRAIL VIOLATION — do not adopt this candidate")


if __name__ == "__main__":
    main()
