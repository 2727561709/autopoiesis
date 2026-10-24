"""RSI Level 1：自产训练信号（iterated DPO 闭环）。
RSI Level 1: self-generated training signal (the iterated-DPO loop).

L0 调的是旋钮；L1 调的是**数据本身**：让学生自己走进环境，在它**当前
会犯错的状态**上做同状态穷举对比，把偏好对长在错误发生的地方，再更新
自己。偏好对分布随策略一起移动——这就是 iterated DPO 与"一次性 DPO"
的区别。
L0 tunes knobs; L1 tunes **the data itself**: the student walks the
environment, we run the same-state exhaustive contrast at the states
where it is CURRENTLY wrong, pairs grow where the mistakes are, then it
updates itself. The pair distribution moves with the policy — that is
what separates iterated DPO from one-shot DPO.

关键工程决策：**每一步都用子进程跑，不在同一进程里叠模型**。
生成数据要载教师、训练要载策略+参考、测量要载学生——任何一个进程同时
载两份 0.5B 模型，在这台 4GB 显存 / 紧张主机内存的机器上都会炸。子进程
让每步都有干净的内存。
Key engineering decision: **every step runs in a subprocess; models are
never stacked in one process.** Data generation loads the teacher,
training loads policy + reference, measurement loads the student — two
0.5B models in one process would blow up on this 4GB / tight-memory
machine. Subprocesses give each step clean memory.

判定与 L0 同一套：显著变好才采纳，守卫失败或显著变差 → 回滚到 parent。
Decisions reuse the L0 rules: adopt only on significant improvement;
guardrail or regression -> roll back to the parent.

默认 **dry-run**：只打印命令，不执行（一次真实周期要跑训练 + 多次探针）。
DRY-RUN by default: print the commands, execute nothing.

用法 / usage:
    py -3.12 -m homeo_lm.selftrain --start runs/dpo-v3 --rounds 3
    py -3.12 -m homeo_lm.selftrain --start runs/dpo-v3 --rounds 3 --run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .rsi import DEFAULT_PROBE_CFG, Ledger, probe_hash

# 复合分：主指标是调制（v2/v3 的核心缺陷），存活为副指标。
# Composite score: modulation is the primary metric (the core deficit in
# v2/v3), survival is secondary. 这是一个**临时**记分板——元探针（对照
# 组实验）才是 RSI 是否有效的最终判据，别把复合分当成目标本身。
# This is a TEMPORARY scoreboard — the meta-probe (control group) is the
# final verdict on whether RSI works; do not worship the composite.
SCORE_WEIGHTS = {"modulation_gap": 1.0, "survival_rate": 1.0,
                 "illegal_rate": -0.5}


def composite_score(metrics: Dict) -> float:
    """复合分（越大越好）。Composite score (higher is better)."""
    return float(sum(w * float(metrics.get(k, 0.0) or 0.0)
                     for k, w in SCORE_WEIGHTS.items()))


def read_probe_report(ckpt: Path) -> Dict:
    """读探针报告。Read the probe report for a checkpoint."""
    p = Path(ckpt) / "probe_report.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def metrics_from_report(report: Dict) -> Dict:
    """报告 → 指标向量。Report -> metric vector."""
    return {
        "modulation_gap": (report.get("modulation") or {}).get(
            "modulation_gap"),
        "survival_rate": (report.get("survival") or {}).get("survival_rate"),
        "illegal_rate": max((report.get("agreement") or {}).get(
            "illegal_rate", 0.0), (report.get("survival") or {}).get(
            "illegal_rate", 0.0)),
        "agreement": (report.get("agreement") or {}).get("agreement"),
    }


def gen_cmd(round_dir: Path, start: Path, pairs: int, teacher: str,
            shape_w: float, policy_frac: float) -> List[str]:
    """第 r 轮：用**当前策略**生成偏好对（on-policy）。
    Round r: generate pairs with the CURRENT policy (on-policy)."""
    round_dir, start = Path(round_dir), Path(start)
    return [sys.executable, "-m", "homeo_lm.gen_dpo_data",
            "--teacher", teacher,
            "--pairs", str(pairs),
            "--shape-w", str(shape_w),
            "--policy", str(start),
            "--policy-frac", str(policy_frac),
            "--out", str(round_dir / "pairs.jsonl")]


def train_cmd(round_dir: Path, start: Path, beta: float,
              epochs: int) -> List[str]:
    """DPO 更新（从当前策略出发，不是从 v2 重来）。
    The DPO update (starting from the current policy, not from v2)."""
    round_dir, start = Path(round_dir), Path(start)
    return [sys.executable, "-m", "homeo_lm.dpo",
            "--data", str(round_dir / "pairs.jsonl"),
            "--ref", str(start),
            "--out", str(round_dir / "adapter"),
            "--beta", str(beta),
            "--epochs", str(epochs)]


def probe_cmd(ckpt: Path, teacher: str, seed_offset: int) -> List[str]:
    """一次独立测量的取样（多次不同 seed-offset → 显著性判定）。
    One independent measurement sample (several offsets -> significance)."""
    return [sys.executable, "-m", "homeo_lm.probe",
            "--ckpt", str(ckpt),
            "--teacher", teacher,
            "--seed-offset", str(seed_offset)]


def round_plan(round_idx: int, start: Path, out: Path, teacher: str,
               pairs: int = 2000, beta: float = 0.1, epochs: int = 1,
               shape_w: float = 0.05, policy_frac: float = 0.34,
               samples: int = 3) -> Dict:
    """一轮 L1 的命令序列。Command sequence for one L1 round."""
    rd = Path(out) / f"round{round_idx}"
    cmds = {
        "gen": gen_cmd(rd, start, pairs, teacher, shape_w, policy_frac),
        "train": train_cmd(rd, start, beta, epochs),
        "probe": [probe_cmd(rd / "adapter", teacher, 1_000 * (k + 1))
                  for k in range(samples)],
        "probe_parent": [probe_cmd(start, teacher, 1_000 * (k + 1))
                         for k in range(samples)],
    }
    return {"round": round_idx, "dir": str(rd), "start": str(start),
            "commands": cmds}


def _run(cmd: Sequence[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def run_rounds(args) -> None:
    """跑若干轮 L1：生成 → 训练 → 测量 → 判定 → 记账。
    Run L1 rounds: generate -> train -> measure -> decide -> record."""
    from .rsi_l0 import decide

    ledger = Ledger(Path(args.ledger))
    hist = ledger.read()
    current = Path(args.start)
    ph = probe_hash(DEFAULT_PROBE_CFG)
    for r in range(1, args.rounds + 1):
        plan = round_plan(r, current, Path(args.out), args.teacher,
                          pairs=args.pairs, beta=args.beta,
                          epochs=args.epochs, samples=args.samples)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if not args.run:
            print(f"DRY-RUN round {r}: 未执行（加 --run 才真跑）"
                  " / nothing executed (add --run)")
            ledger.append({"tag": f"L1-plan-r{r}", "verdict": "plan",
                           "change": {"level": "L1", "round": r},
                           "probe_hash": ph, "ckpt": str(current)})
            continue
        rd = Path(plan["dir"])
        rd.mkdir(parents=True, exist_ok=True)
        _run(plan["commands"]["gen"])
        _run(plan["commands"]["train"])
        cand = rd / "adapter"
        for c in plan["commands"]["probe"]:
            _run(c)
        for c in plan["commands"]["probe_parent"]:
            _run(c)
        cand_scores = [composite_score(metrics_from_report(
            read_probe_report(cand))) for _ in range(args.samples)]
        # parent 的取样来自同一批 seed-offset（配对比较）
        # parent samples use the same offsets (paired comparison)
        par_scores = [composite_score(metrics_from_report(
            read_probe_report(current))) for _ in range(args.samples)]
        rep = read_probe_report(cand)
        guards_ok = True
        if metrics_from_report(rep).get("illegal_rate", 0.0) > 0.2:
            guards_ok = False
        verdict = decide(par_scores, cand_scores, guardrails_ok=guards_ok)
        entry = ledger.append({
            "tag": f"L1-r{r}", "verdict": verdict["verdict"],
            "change": {"level": "L1", "round": r},
            "ckpt": str(cand), "parent": str(current),
            "probe_hash": ph,
            "scores": {"parent": par_scores, "candidate": cand_scores},
            "reason": verdict.get("reason"),
        })
        print(json.dumps(entry, ensure_ascii=False, indent=2, default=str),
              flush=True)
        if verdict["verdict"] == "adopt":
            print(f"ADOPT -> current = {cand}")
            current = cand
        else:
            print(f"{verdict['verdict'].upper()} -> keep {current}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RSI L1：自产训练信号闭环（默认 dry-run）"
                    " / RSI L1: self-generated training signal loop "
                    "(dry-run by default)")
    ap.add_argument("--start", type=str, default="runs/dpo-v3",
                    help="起始适配器 / starting adapter")
    ap.add_argument("--out", type=str, default="runs/selftrain",
                    help="每轮产物目录 / per-round output dir")
    ap.add_argument("--teacher", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=2000,
                    help="每轮偏好对数（含 on-policy 部分）")
    ap.add_argument("--policy-frac", type=float, default=0.34,
                    help="on-policy 状态占比 / on-policy share")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3,
                    help="每版独立测量次数（显著性判定需要 ≥2）"
                         " / independent measurements per version")
    ap.add_argument("--ledger", type=str, default="runs/rsi_ledger.jsonl")
    ap.add_argument("--run", action="store_true",
                    help="真正执行（默认 dry-run）/ actually execute")
    args = ap.parse_args()
    run_rounds(args)


if __name__ == "__main__":
    main()
