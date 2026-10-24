"""RSI L1 的单元测试（不需要模型/显卡）。
Unit tests for RSI L1 (no model / GPU).

测的是"数据从哪来、命令怎么拼、分数怎么算"——真正的训练要子进程跑，
这里只保证编排逻辑正确。
These test where the data comes from, how commands are assembled and
how the score is computed. Real training runs in subprocesses; this only
guarantees the orchestration logic.
"""
import json

import pytest

from homeo_lm.gen_dpo_data import source_shares
from homeo_lm.selftrain import (SCORE_WEIGHTS, composite_score,
                                gen_cmd, metrics_from_report, probe_cmd,
                                read_probe_report, round_plan, train_cmd)


class TestSourceShares:
    """on-policy 数据的配额。On-policy data quotas."""

    def test_without_policy_splits_evenly(self):
        s = source_shares(4000, 0.0)
        assert s["policy"] == 0
        assert s["teacher"] + s["random"] == 4000

    def test_with_policy_takes_its_share(self):
        s = source_shares(3000, 0.34)
        assert s["policy"] == pytest.approx(1020, abs=1)
        assert s["teacher"] + s["random"] + s["policy"] == 3000

    def test_all_policy(self):
        s = source_shares(1000, 1.0)
        assert s["policy"] == 1000 and s["teacher"] == 0 and s["random"] == 0


class TestCommands:
    """命令拼装：on-policy 必须是默认行为，probe 必须带 seed-offset。
    Command assembly: on-policy must be the default, probes must carry a
    seed offset (otherwise significance has no independent samples)."""

    # Windows 会把 "runs/dpo-v3" 规范化成 "runs\dpo-v3"，比较时用 Path
    # Windows normalises "runs/dpo-v3" to "runs\dpo-v3" — compare via Path
    CKPT = str(__import__("pathlib").Path("runs/dpo-v3"))

    def test_gen_uses_the_current_policy(self):
        cmd = gen_cmd("runs/r1", "runs/dpo-v3", 2000, "teacher.pt", 0.05, 0.34)
        assert "--policy" in cmd and self.CKPT in cmd
        assert "--policy-frac" in cmd

    def test_train_starts_from_the_current_policy(self):
        cmd = train_cmd("runs/r1", "runs/dpo-v3", 0.1, 1)
        assert "--ref" in cmd and self.CKPT in cmd
        assert "--data" in cmd

    def test_probe_commands_have_distinct_seed_offsets(self):
        cmds = [probe_cmd("ckpt", "t.pt", 1000 * (k + 1)) for k in range(3)]
        offs = [c[c.index("--seed-offset") + 1] for c in cmds]
        assert len(set(offs)) == 3, "取样必须有独立种子 / samples need seeds"

    def test_round_plan_has_all_steps(self):
        plan = round_plan(1, "runs/dpo-v3", "runs/selftrain", "t.pt",
                          samples=2)
        assert set(plan["commands"]) == {"gen", "train", "probe",
                                        "probe_parent"}
        assert len(plan["commands"]["probe"]) == 2
        assert "round1" in plan["dir"]


class TestScoring:
    """复合分：调制为主，非法率为负。
    Composite score: modulation primary, illegal rate penalised."""

    def test_good_modulation_scores_higher(self):
        good = {"modulation_gap": 0.8, "survival_rate": 0.5,
                "illegal_rate": 0.0}
        bad = {"modulation_gap": -1.0, "survival_rate": 0.0,
               "illegal_rate": 0.0}
        assert composite_score(good) > composite_score(bad)

    def test_illegal_rate_is_penalised(self):
        clean = {"modulation_gap": 0.5, "survival_rate": 0.5,
                 "illegal_rate": 0.0}
        dirty = {"modulation_gap": 0.5, "survival_rate": 0.5,
                 "illegal_rate": 0.4}
        assert composite_score(dirty) < composite_score(clean)
        assert SCORE_WEIGHTS["illegal_rate"] < 0

    def test_missing_metrics_are_treated_as_zero(self):
        assert composite_score({}) == 0.0

    def test_metrics_from_report_extracts_nested_values(self):
        rep = {"modulation": {"modulation_gap": 0.7},
               "survival": {"survival_rate": 0.4, "illegal_rate": 0.1},
               "agreement": {"illegal_rate": 0.2, "agreement": 0.6}}
        m = metrics_from_report(rep)
        assert m["modulation_gap"] == 0.7
        assert m["survival_rate"] == 0.4
        assert m["illegal_rate"] == 0.2      # 取两者较大 / the larger one

    def test_missing_report_file_is_empty(self, tmp_path):
        assert read_probe_report(tmp_path / "nope") == {}


class TestLedgerIntegration:
    """L1 的记账与 L0 用同一本账。L1 shares the ledger with L0."""

    def test_round_plan_is_json_serialisable(self):
        plan = round_plan(2, "runs/a", "runs/out", "t.pt")
        assert json.loads(json.dumps(plan))["round"] == 2
