"""汇总 runs/*/eval_report.log，输出多组横向对比表。

用法：
    python -m homeo_rl.compare            # 全部有评测结果的组
    python -m homeo_rl.compare --runs base scarce curriculum explore abl-wm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

_DEFAULT_RUNS = ("base", "scarce", "curriculum", "explore", "abl-wm")


def _load(run: str, runs_root: Path) -> Dict | None:
    log = runs_root / run / "eval_report.log"
    if not log.exists():
        return None
    try:
        return json.loads(log.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def summarize(run: str, rep: Dict) -> Dict:
    f = rep["foraging"]
    d = rep["direction_modulation"]
    approach = d["food_approach_rate"]
    return {
        "run": run,
        "survival": rep["survival"]["survival_rate"],
        "final_energy": rep["survival"]["mean_final_energy"],
        "final_integrity": rep["survival"]["mean_final_integrity"],
        "near_visible": f["trained"]["near_visible"]["success_rate"],
        "far_search": f["trained"]["far_search"]["success_rate"],
        "far_random": f["random_baseline"]["far_search"]["success_rate"],
        "approach_mean": sum(approach.values()) / len(approach),
        "satiated_approach": d["satiated_approach_rate"],
        "hazard_entry": d["hazard_entry_rate"],
    }


_COLS = [
    ("run", "组名", "<10"),
    ("survival", "存活", ">7"),
    ("near_visible", "追击(可见)", ">10"),
    ("far_search", "搜索(视野外)", ">12"),
    ("far_random", "随机基线", ">8"),
    ("approach_mean", "定向均值", ">8"),
    ("satiated_approach", "饱腹接近", ">8"),
    ("hazard_entry", "入险率", ">7"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总评测对比表")
    ap.add_argument("--runs", nargs="+", default=list(_DEFAULT_RUNS))
    ap.add_argument("--root", type=str, default="runs")
    args = ap.parse_args()

    rows: List[Dict] = []
    for run in args.runs:
        rep = _load(run, Path(args.root))
        if rep is not None:
            rows.append(summarize(run, rep))

    if not rows:
        print("暂无评测结果")
        return

    header = "".join(f"{label:{fmt}} " for _, label, fmt in _COLS)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for key, _, fmt in _COLS:
            val = row[key]
            cell = val if isinstance(val, str) else f"{val:.2f}"
            line += f"{cell:{fmt}} "
        print(line)

    out = Path(args.root) / "round_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
