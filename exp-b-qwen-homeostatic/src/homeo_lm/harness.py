"""实验 Harness：一条命令跑完 A–E 对照组 × 多 seed 网格。
Experiment harness: run the A-E control grid x seeds with one command.

前面四层各自闭环，但**没有一层回答过那个问题**：

> 指标涨了，是因为"自我改进有效"，还是因为**同等算力下闷头多训也会涨**？

`metaexp` 只给出计划与判定，harness 负责**真的把它跑完并汇总**：

- **网格**：`{A,B,C,D,E} × seeds`，每个格子**同等算力预算**（`equal_compute`
  会核对，不等则整轮作废——这一点不可商量，否则比较没有意义）
- **步骤即子进程**：复用已有 CLI（生成 / 训练 / 测量），不在同一进程叠模型
- **断点续跑**：`--resume` 跳过已完成的 `(组, seed)`，长跑中断不用重来
- **汇总落盘**：`runs/harness_results.jsonl`，一行一个格子
- **判定**：`metaexp.verdict` —— **超可加性** `D > A` 且 `D > B` 且 `D > C`

**默认 dry-run**：只打印计划与命令（`--run` 才真跑）。真实网格要跑
多轮训练 + 多次探针，时机由人决定。
DRY-RUN by default: print the plan and commands (`--run` to execute).

用法 / usage:
    py -3.12 -m homeo_lm.harness --groups ABCDE --seeds 0,1,2
    py -3.12 -m homeo_lm.harness --groups ABCDE --seeds 0,1,2 --run
    py -3.12 -m homeo_lm.harness --groups DE --seeds 0,1,2 --run --resume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .metaexp import DEFAULT_BUDGET, GROUPS, cost, equal_compute, verdict
from .rsi import DEFAULT_PROBE_CFG, Ledger, probe_hash

# 每格训练大约的 update 数（同等算力 → 同值）/ roughly how many updates
# per cell (equal compute -> equal value)
ROUNDS_PER_CELL = 3


def run_key(group: str, seed: int) -> str:
    """格子的唯一标识（断点续跑靠它）。Cell id (resume key)."""
    return f"{group}-s{seed}"


def job_dir(root: str, group: str, seed: int) -> Path:
    return Path(root) / run_key(group, seed)


def job_stages(group: str, outdir: Path, start: Path, teacher: str) -> List[Dict]:
    """一个格子要跑的命令序列（**全部走子进程**）。
    The command sequence for one cell (all in subprocesses).

    组的定义（这是可审计的表格，不是散在代码里的 if）：
    Group definitions (an auditable table, not conditionals scattered
    through the code):

    | 组 | 步骤 | 回答什么 |
    |---|---|---|
    | A | 固定数据训一轮 | 算力本身能买多少性能（对照） |
    | B | L0 调旋钮 | 自适应调参值不值 |
    | C | L1 自产数据 | 自产信号值不值 |
    | D | C 再 B | 两者是否互补（**超可加性**） |
    | E | D + L2 | 改学习规则还有多少增量 |
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    levels = GROUPS[group]["levels"]
    stages: List[Dict] = []
    if not levels:                      # A：固定课程，一直训 / fixed curriculum
        stages.append({
            "name": "train-fixed",
            "cmd": [sys.executable, "-m", "homeo_lm.dpo",
                    "--data", "data/dpo_pairs.jsonl",
                    "--ref", str(start),
                    "--out", str(d / "adapter")],
        })
    if "L1" in levels:                  # C/D/E：先调数据 / data first
        stages.append({
            "name": "l1-selftrain",
            "cmd": [sys.executable, "-m", "homeo_lm.selftrain",
                    "--start", str(start),
                    "--rounds", str(ROUNDS_PER_CELL),
                    "--ledger", str(d / "l1_ledger.jsonl"),
                    "--run"],
        })
    if "L0" in levels:                  # B/D/E：再调旋钮 / then knobs
        stages.append({
            "name": "l0-cycle",
            "cmd": [sys.executable, "-m", "homeo_lm.rsi", "--cycle",
                    "--ckpt", str(start),
                    "--out", str(d / "l0_ledger.jsonl"),
                    "--run-trials"],
        })
    if "L2" in levels:                  # E：改学习规则 / then the rule
        stages.append({
            "name": "l2-diagnose",
            "cmd": [sys.executable, "-m", "homeo_lm.rsi", "--level", "2",
                    "--out", str(d / "l2_ledger.jsonl"),
                    "--holdout", "--seeds", "3"],
        })
    return stages


def grid_plan(groups: Sequence[str], seeds: Sequence[int], start: Path,
              teacher: str, budget: Optional[Dict] = None) -> Dict:
    """整个网格的计划 + 算力相等性核对。
    The plan for the whole grid plus the equal-compute check."""
    b = dict(DEFAULT_BUDGET if budget is None else budget)
    jobs = []
    for g in groups:
        for s in seeds:
            d = job_dir("runs/harness", g, s)
            jobs.append({"group": g, "seed": s, "key": run_key(g, s),
                         "dir": str(d),
                         "stages": job_stages(g, d, start, teacher)})
    plans = [{"group": g, "cost": cost(b, 1)} for g in groups]
    return {"budget": b, "seeds": list(seeds), "jobs": jobs,
            "equal_compute": equal_compute(plans),
            "note": "A=control, B=L0, C=L1, D=L0+L1 (superadditivity), "
                    "E=+L2"}


# ------------------------------------------------------------ 结果：读与写
def read_results(path: str) -> List[Dict]:
    """读结果文件（坏行跳过并计数，不让一行坏数据毁掉整轮）。
    Read the results file (bad lines are skipped, not fatal)."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def completed_keys(path: str) -> set:
    """已完成的格子（断点续跑用）。Cells already done (for --resume)."""
    return {r.get("key") for r in read_results(path) if r.get("key")}


def final_ckpt(outdir: Path, start: Path) -> str:
    """一格的最终产物：优先本格训练出的适配器，否则保留 parent。
    The cell's final artefact: its own trained adapter if one exists,
    else the parent is kept.

    为什么看账本：B 组的 L0 周期会跑多个试验，只有**被采纳**的那一版才是
    结果；一个都没采纳 → 原地保留 parent（这正是"拒绝"该有的样子）。
    Why the ledger: group B's L0 cycle runs several trials, and only an
    ADOPTED one counts; if none was adopted the parent stays — which is
    exactly what a rejection should look like.
    """
    d = Path(outdir)
    own = d / "adapter"
    if own.exists():
        return str(own)
    for name in ("l0_ledger.jsonl", "l1_ledger.jsonl", "l2_ledger.jsonl"):
        p = d / name
        if not p.exists():
            continue
        rows = Ledger(p).read()
        for e in reversed(rows):
            if e.get("verdict") == "adopt" and e.get("ckpt"):
                return str(e["ckpt"])
    return str(start)


def append_result(path: str, row: Dict) -> Dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def execute_job(job: Dict, teacher: str, samples: int = 3,
                device: Optional[str] = None,
                results_path: Optional[str] = None) -> Dict:
    """跑完一个格子：命令 → 最终产物 → 多次独立测量 → 记账。
    Run one cell: commands -> final artefact -> independent
    measurements -> record."""
    from .rsi import measure_ckpt

    for st in job["stages"]:
        print("$", " ".join(st["cmd"]), flush=True)
        subprocess.run(st["cmd"], check=True)

    ckpt = final_ckpt(Path(job["dir"]), Path(job["start"]))
    scores = measure_ckpt(Path(ckpt), teacher, samples=samples,
                          device=device)
    row = {"key": job["key"], "group": job["group"], "seed": job["seed"],
           "ckpt": ckpt, "scores": scores,
           "score": (sum(scores) / len(scores)) if scores else None,
           "probe_hash": probe_hash(DEFAULT_PROBE_CFG),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if results_path:
        append_result(results_path, row)
    return row


def run_grid(args) -> None:
    """网格入口（dry-run 只打印）。Grid entry point (dry-run prints)."""
    seeds = [int(s) for s in args.seeds.split(",")]
    groups = list(args.groups)
    plan = grid_plan(groups, seeds, Path(args.start), args.teacher,
                     budget={"trials": args.max_trials,
                             "train_steps": DEFAULT_BUDGET["train_steps"],
                             "probe_runs": args.samples})
    print(json.dumps({"budget": plan["budget"], "seeds": plan["seeds"],
                      "equal_compute": plan["equal_compute"],
                      "jobs": [{"key": j["key"], "dir": j["dir"],
                                "stages": [s["name"] for s in j["stages"]]}
                               for j in plan["jobs"]]},
                     ensure_ascii=False, indent=2))
    if not plan["equal_compute"]["equal"]:
        print("ABORT: 各组算力不等 — 比较无效"
              " / compute differs across groups: comparison is void")
        return
    if not args.run:
        for j in plan["jobs"]:
            for s in j["stages"]:
                print("CMD", j["key"], ":", " ".join(s["cmd"]))
        print("DRY-RUN: 未执行（加 --run 才真跑）"
              " / nothing executed (add --run)")
        return

    done = completed_keys(args.out) if args.resume else set()
    for j in plan["jobs"]:
        if j["key"] in done:
            print("skip (done):", j["key"], flush=True)
            continue
        j["start"] = args.start
        row = execute_job(j, args.teacher, samples=args.samples,
                          device=args.device, results_path=args.out)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rows = read_results(args.out)
    agg: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("score") is not None:
            agg.setdefault(r["group"], []).append(r["score"])
    v = verdict(agg, noise=args.noise)
    print(json.dumps({"verdict": v}, ensure_ascii=False, indent=2,
                     default=str))


def main() -> None:  # pragma: no cover - CLI 入口 / CLI entry
    ap = argparse.ArgumentParser(
        description="实验 Harness：跑 A–E 对照组网格并判定 RSI 是否真的有效"
                    " / experiment harness: run the A-E grid and judge "
                    "whether RSI actually pays")
    ap.add_argument("--groups", type=str, default="ABCDE")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--start", type=str, default="runs/sft-lora-v2",
                    help="所有组的训练起点（必须相同，否则无法归因）"
                         " / shared training start (must be identical)")
    ap.add_argument("--teacher", type=str,
                    default="../exp-a-homeostatic-rl/runs/curriculum/"
                            "checkpoint.pt")
    ap.add_argument("--out", type=str, default="runs/harness_results.jsonl")
    ap.add_argument("--samples", type=int, default=3,
                    help="每格独立测量次数 / measurements per cell")
    ap.add_argument("--max-trials", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.02,
                    help="判定门槛：差值必须超过它才算证据"
                         " / margin a difference must clear")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--run", action="store_true",
                    help="真跑（默认 dry-run）/ execute for real")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已完成的格子 / skip completed cells")
    run_grid(ap.parse_args())


if __name__ == "__main__":
    main()
